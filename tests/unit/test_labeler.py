"""prl/algo/labeler.py: the steps as the judge sees them, the judge prompt, answer parsing, and labeling fallbacks."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from prl.algo import labeler, llm_client
from prl.transforms.core import Trajectory

_real_post = llm_client._post  # the judge fixture replaces it


class CharTokenizer:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _sample(prompt: str, parts: list[tuple[str, int]], node_id: int, **md):
    """parts: (text, trained) spans of the response, in order."""
    response = [ord(c) for text, _ in parts for c in text]
    mask = [t for text, t in parts for _ in text]
    return SimpleNamespace(tokens=[ord(c) for c in prompt] + response, response_length=len(response),
                           loss_mask=mask, metadata={"leaf": {"node_id": node_id}, **md})


def _traj(*samples):
    return Trajectory.of(list(samples), CharTokenizer())


def test_render_clips_thinking_and_strips_observations():
    after_summary = _sample("P2", [("C", 1), ("o3 ", 0)], node_id=9)
    before = _sample("TASK", [("<think>t</think>A", 1), (" o1", 0), ("B", 1), ("o2", 0)], node_id=4)
    task, steps = labeler.render(_traj(after_summary, before))
    assert task == "TASK"  # the first leaf's prompt
    assert steps[0] == labeler.Rendered("<think>\nt\n</think>\nA", "o1")
    assert [s.observation for s in steps[1:]] == ["o2", "o3"]


def test_clip_and_prompt():
    assert labeler.clip("abcdefghij", 6) == "abcd\n[... 4 characters omitted ...]\nij"
    steps = [labeler.Rendered("a", "x"), labeler.Rendered("b", "")]
    for style in ("paper", "short"):
        prompt = labeler.build_prompt("T", steps, 0.0, style)
        assert "### Step 2\n[model]\nb\n[received]\n(nothing)" in prompt and "tests failed" in prompt
    assert "A maximum of 1 steps" in labeler.build_prompt("T", steps, 0.0)
    assert [labeler.step_cap(n) for n in (1, 3, 4, 10, 17)] == [1, 1, 1, 3, 5]


def test_parse_answer():
    content = '<think>{"critical_steps": [9]}</think>\nSure: {"critical_steps": [3, 1, 3, 7], "reasons": {"1": "plan"}}'
    assert labeler.parse_answer(content, 5) == ([1, 3], {"1": "plan"})
    for bad in ("no json", '{"critical_steps": []}', '{"critical_steps": [6]}'):
        with pytest.raises(ValueError):
            labeler.parse_answer(bad, 5)
    with pytest.raises(ValueError, match="cap"):
        labeler.parse_answer('{"critical_steps": [1, 2]}', 5, cap=1)



def test_parse_grades():
    content = '<think>{"good": [9], "bad": []}</think> {"good": [1, 3], "bad": [2, 9], "reasons": {"2": "wrong path"}}'
    assert labeler.parse_grades(content, 5, cap=2) == ({1, 3}, {2}, {"2": "wrong path"})
    assert labeler.parse_grades('{"good": [], "bad": []}', 5, cap=1) == (set(), set(), {})  # all neutral: valid
    for bad in ("no json", '{"critical_steps": [1]}', '{"good": [1], "bad": [1]}', '{"good": [1, 2], "bad": []}',
                '{"good": [], "bad": [1, 2]}'):
        with pytest.raises(ValueError):
            labeler.parse_grades(bad, 5, cap=1)
    assert "Mark\nat most 3 steps good and at most 3 steps bad" in labeler.build_prompt("T", [labeler.Rendered("a", "o")] * 10, 0.0, "grade")

@pytest.fixture
def judge(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("secret\n")
    monkeypatch.setenv("PRL_JUDGE_API_BASE", "http://judge/v1")
    monkeypatch.setenv("PRL_JUDGE_KEY_FILE", str(key))
    monkeypatch.setenv("PRL_JUDGE_MODEL", "m")
    monkeypatch.setenv("PRL_JUDGE_ATTEMPTS", "2")
    sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: sleep(0))  # no retry backoff
    calls = []

    def answer(contents):
        async def post(cfg, messages):
            calls.append((cfg["key"], messages[-1]["content"]))
            content = contents[min(len(calls), len(contents)) - 1]
            if isinstance(content, Exception):
                raise content
            return {"choices": [{"message": {"content": content}}], "usage": {}}

        monkeypatch.setattr(llm_client, "_post", post)
        return calls

    return answer


def test_label_retries_then_writes_record(judge, tmp_path):
    calls = judge([RuntimeError("503"), '{"critical_steps": [2]}'])
    sample = _sample("T", [("A", 1), ("o", 0), ("B", 1)], node_id=1, reward=1.0, trial_dir=str(tmp_path))
    critical = asyncio.run(labeler.label(_traj(sample)))
    assert critical == {1} and len(calls) == 2 and calls[0][0] == "secret"
    record = json.loads((tmp_path / "atlas_labels.json").read_text())
    assert record["critical_steps"] == [2] and not record["fallback"] and "503" in record["errors"][0]
    assert "tests passed" in record["prompt"] and "secret" not in json.dumps(record)



def test_grade_writes_record(judge, tmp_path):
    calls = judge(['{"good": [1], "bad": [1]}', '{"good": [1], "bad": [2], "reasons": {"2": "x"}}'])
    sample = _sample("T", [("A", 1), ("o", 0), ("B", 1)], node_id=1, reward=1.0, trial_dir=str(tmp_path))
    assert asyncio.run(labeler.grade(_traj(sample))) == ({0}, {1}) and len(calls) == 2  # both good and bad: retried
    record = json.loads((tmp_path / "atlas_labels.json").read_text())
    assert (record["good_steps"], record["bad_steps"], record["prompt_style"]) == ([1], [2], "grade")
    assert "- bad: a step with a mistake" in record["prompt"] and "x" == record["reasons"]["2"]

def test_grade_quoted(judge, tmp_path):
    calls = judge(['{"good": [1], "bad": [2], "quotes": {"2": "rm -rf /x", "1": "not bad", "x": "bad key", "3": 4}}'])
    sample = _sample("T", [("A", 1), ("o", 0), ("B", 1), ("o", 0), ("C", 1)], node_id=1, reward=1.0,
                     trial_dir=str(tmp_path))
    good, bad, quotes = asyncio.run(labeler.grade_quoted(_traj(sample)))
    assert (good, bad, quotes) == ({0}, {1}, {1: "rm -rf /x"}) and len(calls) == 1  # quotes of bad steps only
    record = json.loads((tmp_path / "atlas_labels.json").read_text())
    assert record["prompt_style"] == "grade_quote" and record["quotes"] == {"2": "rm -rf /x", "1": "not bad"}
    assert "quote its mistake: copy, character for character" in record["prompt"]
    assert labeler.parse_quotes('{"good": [], "bad": [1]}') == {}  # no quotes: the whole step falls back


def test_label_waits_out_rate_limits(judge, monkeypatch):
    import httpx

    limited = httpx.HTTPStatusError("429", request=None, response=httpx.Response(429))
    calls = judge([limited] * 5 + ['{"critical_steps": [1]}'])
    monkeypatch.setattr(llm_client.random, "uniform", lambda a, b: 0)
    critical = asyncio.run(labeler.label(_traj(_sample("T", [("A", 1)], node_id=1))))
    assert critical == {0} and len(calls) == 6  # 5 rate limits do not use up the 2 attempts


@pytest.mark.parametrize("down", ["connection", "503"])
def test_label_waits_out_an_unavailable_judge(judge, monkeypatch, down):
    import httpx

    error = (httpx.ConnectError("connection refused") if down == "connection"
             else httpx.HTTPStatusError("503", request=None, response=httpx.Response(503)))
    calls = judge([error] * 5 + ['{"critical_steps": [1]}'])
    monkeypatch.setattr(llm_client.random, "uniform", lambda a, b: 0)
    record = {}
    answer = asyncio.run(llm_client.ask([{"role": "user", "content": "x"}], str, record=record))
    assert answer == '{"critical_steps": [1]}' and len(calls) == 6 and record["unavailable"] == 5  # no attempt used


def test_an_outage_longer_than_the_wait_uses_the_attempts(judge, monkeypatch):
    import httpx

    calls = judge([httpx.ConnectError("connection refused")] * 5 + ['{"critical_steps": [1]}'])
    monkeypatch.setattr(llm_client, "RATE_LIMIT_WAIT", 0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(llm_client.asyncio, "sleep", lambda s: real_sleep(0))
    record = {}
    assert asyncio.run(llm_client.ask([{"role": "user", "content": "x"}], str, record=record)) is None
    assert len(calls) == int(os.environ["PRL_JUDGE_ATTEMPTS"]) and record["unavailable"] == 0


def test_a_refused_prompt_is_not_asked_again(judge):
    import httpx

    body = '{"error":{"message":"Invalid prompt: your prompt was flagged as potentially violating our usage policy."}}'
    refused = httpx.HTTPStatusError("502", request=None, response=httpx.Response(502, text=body))
    calls = judge([refused, '{"critical_steps": [1]}'])
    record = {}
    assert asyncio.run(llm_client.ask([{"role": "user", "content": "x"}], str, record=record)) is None
    assert len(calls) == 1 and record["refused"] == 1 and "flagged" in record["errors"][0]


@pytest.mark.parametrize("refusals", [1, 2, 3])
def test_a_refused_prompt_goes_to_the_next_model(judge, monkeypatch, tmp_path, refusals):
    """GPT's prompt filter refused 28-84 of 256 trials per step late in one training run, so a refused prompt goes to
    the next model (gpt-6.1-sol, then gpt-6-astra, then gpt-5.6-sol); the labels name the judge that answered."""
    import httpx

    monkeypatch.setenv("PRL_JUDGE_FALLBACK_MODELS", " astra , sol56 ")
    body = '{"error":{"message":"Invalid prompt: your prompt was flagged as potentially violating our usage policy."}}'
    asked = []

    async def post(cfg, messages):
        asked.append(cfg["model"])
        if len(asked) <= refusals:
            raise httpx.HTTPStatusError("502", request=None, response=httpx.Response(502, text=body))
        return {"choices": [{"message": {"content": '{"critical_steps": [1]}'}}], "usage": {}, "model": cfg["model"]}
    monkeypatch.setattr(llm_client, "_post", post)
    sample = _sample("T", [("A", 1)], node_id=1, reward=1.0, trial_dir=str(tmp_path))
    critical = asyncio.run(labeler.label(_traj(sample)))
    record = json.loads((tmp_path / "atlas_labels.json").read_text())
    chain = ["m", "astra", "sol56"]
    assert asked == chain[:min(refusals + 1, 3)]  # a refusal is not asked again of the same model
    assert record["refused_by"] == chain[:refusals] and record["refused"] == refusals
    if refusals < 3:
        assert critical == {0} and record["model"] == chain[refusals] == record["served_model"]
    else:  # every model refused: the trial keeps all steps, as without fallbacks
        assert critical is None and record["fallback"] and record["model"] == "sol56"


def test_the_judge_is_asked_the_same_question_again(judge):
    calls = judge(["nothing useful", '{"critical_steps": [1]}'])  # ATLAS: no feedback on a rejected answer
    assert asyncio.run(labeler.label(_traj(_sample("T", [("A", 1)], node_id=1)))) == {0}
    assert len(calls) == 2 and calls[0] == calls[1]


def test_label_falls_back_when_unparsable(judge):
    judge(["nothing useful"])
    critical = asyncio.run(labeler.label(_traj(_sample("T", [("A", 1)], node_id=1))))
    assert critical is None


def test_extra_body_reaches_the_request_and_the_record(judge, monkeypatch, tmp_path):
    import httpx

    sent = []

    class Client:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json, headers):
            sent.append(json)
            return httpx.Response(200, json={"model": "m-snapshot", "choices": [
                {"message": {"content": '{"critical_steps": [1]}'}, "finish_reason": "stop"}]},
                request=httpx.Request("POST", url))

    monkeypatch.setattr(llm_client, "_post", _real_post)
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setenv("PRL_JUDGE_EXTRA_BODY", '{"enable_thinking": true}')
    critical = asyncio.run(labeler.label(_traj(_sample("T", [("A", 1)], node_id=1, trial_dir=str(tmp_path)))))
    assert critical == {0} and sent[0]["enable_thinking"] is True and sent[0]["model"] == "m"
    record = json.loads((tmp_path / "atlas_labels.json").read_text())
    assert record["extra_body"] == {"enable_thinking": True}
    assert (record["served_model"], record["finish_reason"]) == ("m-snapshot", "stop")


def test_extra_body_must_be_an_object(judge, monkeypatch):
    monkeypatch.setenv("PRL_JUDGE_EXTRA_BODY", "[1]")
    with pytest.raises(ValueError, match="JSON object"):
        llm_client.config()
    monkeypatch.delenv("PRL_JUDGE_EXTRA_BODY")
    assert llm_client.config()["extra_body"] == {}


def test_prompt_style_is_checked(monkeypatch):
    monkeypatch.setenv("PRL_JUDGE_PROMPT", "other")
    with pytest.raises(ValueError, match="paper or short"):
        labeler.prompt_style()
