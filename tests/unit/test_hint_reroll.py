"""transforms/hint_reroll.py: the judge's fork and hint, the answer checks, which trials branch."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from prl.transforms import Unanswered, core
from transforms import hint_reroll as module

PROMPT = "<|im_start|>user\nTASK<|im_end|>\n<|im_start|>assistant\n<think>\n"


class Tok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _turn(thinking, command):
    return f"{thinking}\n</think>\n\n{json.dumps(command)}<|im_end|>"


def _message(text):
    return f"\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def _trial(trial_dir, reward=0.0):
    parts = [(_turn("look around", {"commands": [{"keystrokes": "ls\n"}]}), 1), (_message("a b"), 0),
             (_turn("convert it", {"commands": [{"keystrokes": "python3 conv.py\n"}]}), 1), (_message("error"), 0),
             (_turn("done", {"commands": [], "task_complete": True}), 1)]
    response = [ord(c) for text, _ in parts for c in text]
    return [SimpleNamespace(tokens=[ord(c) for c in PROMPT] + response, response_length=len(response),
                            loss_mask=[t for text, t in parts for _ in text], reward=reward, rollout_id=7,
                            rollout_log_probs=[0.5] * len(response), train_metadata=None,
                            metadata={"leaf": {"node_id": 0}, "reward": reward, "trial_dir": trial_dir})]


@pytest.fixture
def judge(monkeypatch):
    """Run hint_reroll with the judge's answers given in order; returns the prompts it was asked."""
    for key in ("PRL_TRANSFORM_ARGS", "PRL_TRANSFORM_MAX_FAILURES",
                "PRL_TRANSFORM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(core, "_failures", 0)
    monkeypatch.setenv("PRL_TRANSFORM", "hint_reroll")
    asked, answers = [], []

    async def ask(prompt, check=None):  # as prl.transforms.ask: a refused answer is asked again
        asked.append(prompt)
        errors = []
        for content in answers:
            try:
                return check(content)
            except ValueError as exc:
                errors.append(str(exc))
        raise Unanswered(f"no acceptable answer: {errors}")
    monkeypatch.setattr(module, "ask", ask)

    def run(samples, *contents, args=None):
        answers[:] = contents
        if args is not None:
            monkeypatch.setenv("PRL_TRANSFORM_ARGS", json.dumps(args))
        out = asyncio.run(core.run(samples, Tok()))
        return out[0].metadata["prl_transform"]
    run.asked = asked
    return run


def test_a_failed_trial_branches_from_the_judges_step_with_its_hint(judge, tmp_path):
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier" / "test-stdout.txt").write_text("FAILED test_output - expected 3 columns\n")
    record = judge(_trial(str(tmp_path)), '{"step": 2, "hint": "Check what the input file actually contains."}')
    [anchor] = record["reroll"]["anchors"]
    assert (anchor["step"], anchor["n"], anchor["credit"]) == (1, 2, "verified")
    assert anchor["hint"] == "Hint: Check what the input file actually contains."
    assert record["chain"][0]["record"]["hint_reroll"]["step"] == 2
    [prompt] = judge.asked
    assert "### Step 3" in prompt and "The task's tests failed" in prompt
    assert "expected 3 columns" in prompt and "at most 15 words" in prompt and "where to look" in prompt


def test_passed_and_unhelpful_trials_are_left_alone(judge, tmp_path):
    record = judge(_trial(str(tmp_path), reward=1.0))
    assert "reroll" not in record and not judge.asked  # no judge call for a passed trial
    record = judge(_trial(str(tmp_path)), '{"step": null, "hint": null}')
    assert "reroll" not in record and record["chain"][0]["record"]["hint_reroll"]["why"].startswith("the judge found")
    record = judge(_trial(str(tmp_path)), "no idea")
    assert "reroll" not in record and record["chain"][0]["status"] == "skipped"


def test_a_refused_answer_is_asked_again(judge, tmp_path):
    record = judge(_trial(str(tmp_path)), '{"step": 1, "hint": "Look again."}',
                   '{"step": 3, "hint": "Run `cat data.csv` first."}', '{"step": "3", "hint": "Re-read the task."}',
                   args={"n": 1, "test_chars": 0})
    [anchor] = record["reroll"]["anchors"]
    assert (anchor["step"], anchor["n"], anchor["hint"]) == (2, 1, "Hint: Re-read the task.")


@pytest.mark.parametrize("answer, why", [
    ('{"step": 1, "hint": "Look again."}', "from 2 to 3"),
    ('{"step": 4, "hint": "Look again."}', "from 2 to 3"),
    ('{"step": true, "hint": "Look again."}', "from 2 to 3"),
    ('{"step": 2, "hint": ""}', "one short sentence"),
    ('{"step": 2, "hint": "Look again.\\nAnd again."}', "one line"),
    ('{"step": 2, "hint": "' + "word " * 26 + '"}', "26 words"),
    ('{"step": 2, "hint": "Use cut -d, | sort."}', "|"),
    ('{"step": 2, "hint": "Check $HOME first."}', "$"),
    ('{"step": 2, "hint": "Write it > out.txt."}', ">"),
    ("the answer is step 2", "no JSON"),
])
def test_answer_checks(answer, why):
    with pytest.raises(ValueError, match=why.replace("$", r"\$").replace("|", r"\|")):
        module.parse_answer(answer, 3, 25)


def test_accepted_answers():
    assert module.parse_answer('<think>step 1?</think>{"step": null}', 3, 25) == {"step": None, "hint": None}
    assert module.parse_answer('Answer: {"step": 3, "hint": "  Re-read   the task. "}', 3, 25) == {
        "step": 3, "hint": "Re-read the task."}


@pytest.mark.parametrize("args", [{"credit": "group"}, {"max_hint_words": 0}])
def test_bad_params(args):
    traj = SimpleNamespace(reward=0.0, samples=[], steps=[], record={}, trial_dir=None)
    with pytest.raises(ValueError):
        asyncio.run(module.hint_reroll(traj, **args))


def test_the_last_json_object_is_read_past_other_braces():
    content = 'Step {3} looked off. Answer: {"step": 3, "hint": "Re-read the task."}'
    assert module.parse_answer(content, 3, 25) == {"step": 3, "hint": "Re-read the task."}
