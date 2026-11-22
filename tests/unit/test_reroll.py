"""Reroll: ``step.reroll()`` in a transform (prl/transforms/core.py), the agent that replays the first turns
(prl/rollout/reroll_agent.py), the rollout worker that runs it (prl/rollout/harbor_training.py ``_reroll``)
and its metrics; no GPU, sandbox or Miles needed (fake tokenizer, fake generator, fake Sample status)."""
from __future__ import annotations

import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from prl.rollout import branching, harbor_training
from prl.rollout.rollout_metrics import compute
from prl.transforms import core

PROMPT = "<|im_start|>user\nTASK<|im_end|>\n<|im_start|>assistant\n<think>\n"


class Tok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _turn(thinking, command):
    return f"{thinking}\n</think>\n\n{json.dumps(command, indent=2)}<|im_end|>"


def _message(text, last=False):
    return f"\n<|im_start|>user\n{text}<|im_end|>" + ("" if last else "\n<|im_start|>assistant\n<think>\n")


def _trial(trial_dir="/trials/t1", reward=1.0):
    parts = [(_turn("look around", {"commands": [{"keystrokes": "ls\n"}]}), 1), (_message("New Terminal Output:\n\na b"), 0),
             (_turn("list again", {"commands": [{"keystrokes": "ls -la\n"}]}), 1), (_message("New Terminal Output:\n\nc"), 0),
             (_turn("done", {"commands": [], "task_complete": True}), 1)]
    response = [ord(c) for text, _ in parts for c in text]
    return [SimpleNamespace(tokens=[ord(c) for c in PROMPT] + response, response_length=len(response),
                            loss_mask=[t for text, t in parts for _ in text], reward=reward, rollout_id=7,
                            rollout_log_probs=[0.5] * len(response), train_metadata=None,
                            metadata={"leaf": {"node_id": 0}, "reward": reward, "trial_dir": trial_dir})]


@pytest.fixture
def use(monkeypatch):
    for key in ("PRL_TRANSFORM_ARGS", "PRL_TRANSFORM_MAX_FAILURES",
                "PRL_TRANSFORM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(core, "_failures", 0)
    fns = {}
    monkeypatch.setattr(core, "_resolve", lambda name: fns[name])

    def select(fn):
        fns["t"] = fn
        monkeypatch.setenv("PRL_TRANSFORM", "t")
    return select


def _run(samples):
    return asyncio.run(core.run(samples, Tok()))


def _record(samples):
    return samples[0].metadata["prl_transform"]


# --- the transform's side --------------------------------------------------------------------------------------------

def test_reroll_request_has_the_turns_before_it_as_messages(use, tmp_path):
    use(lambda traj: traj.steps[2].reroll())
    out = _run(_trial(str(tmp_path)))
    [request] = _record(out)["reroll"]["anchors"]
    assert request["step"] == 2 and request["instruction"] == "TASK" and request["trial_dir"] == str(tmp_path)
    assert (request["n"], request["credit"]) == (1, "member")
    assert [t["step"] for t in request["turns"]] == [0, 1]
    assert request["turns"][0]["reasoning"] == "look around"
    assert json.loads(request["turns"][1]["content"]) == {"commands": [{"keystrokes": "ls -la\n"}]}
    assert request["turns"][0]["message"] == "New Terminal Output:\n\na b"
    first_two = sum(len(t) for t, _ in [(_turn("look around", {"commands": [{"keystrokes": "ls\n"}]}), 1),
                                        (_message("New Terminal Output:\n\na b"), 0),
                                        (_turn("list again", {"commands": [{"keystrokes": "ls -la\n"}]}), 1)])
    assert request["context_tokens"] == len(PROMPT) + first_two
    # the rerolled step's first response token: after the second turn's terminal message
    assert request["token_start"] == first_two + len(_message("New Terminal Output:\n\nc"))
    assert _record(out)["changed"] and _record(out)["chain"][0]["reroll_steps"] == [2]
    assert out[0].tokens == _trial()[0].tokens  # the trial itself trains as generated


def test_a_hint_and_verified_credit_go_into_the_request(use):
    use(lambda traj: traj.steps[2].reroll(2, "verified", hint="Hint: list the hidden files too."))
    [request] = _record(_run(_trial()))["reroll"]["anchors"]
    assert (request["n"], request["credit"], request["hint"]) == (2, "verified", "Hint: list the hidden files too.")
    use(lambda traj: traj.steps[2].reroll())
    assert _record(_run(_trial()))["reroll"]["anchors"][0]["hint"] is None


def test_a_rewrite_before_the_rerolled_step_reaches_only_the_reroll(use):
    def fn(traj):
        traj.steps[1].edit_thinking("list again", "look closer")
        traj.steps[0].mask()
        traj.steps[2].weight = -1.0
        traj.steps[2].reroll()
    use(fn)
    out = _run(_trial())
    [request] = _record(out)["reroll"]["anchors"]
    assert request["turns"][1]["reasoning"] == "look closer" and request["rewritten_steps"] == [1]
    s, original = out[0], _trial()[0]
    assert s.tokens == original.tokens and _record(out)["tokens_changed"] is False  # the rewrite was not applied here
    first = len(_turn("look around", {"commands": [{"keystrokes": "ls\n"}]}))
    assert not any(s.loss_mask[:first])  # masks and weights stay on the trial
    assert s.train_metadata["step_weights"] == [1.0, -1.0]


def test_a_deleted_turn_is_left_out_of_the_replay(use):
    def fn(traj):
        traj.steps[0].delete()
        traj.steps[2].reroll()
    use(fn)
    [request] = _record(_run(_trial()))["reroll"]["anchors"]
    assert [t["step"] for t in request["turns"]] == [1]


def test_several_steps_branch_n_times_each(use):
    """Several anchors in one trial: in step order, each with its own replay, n and credit."""
    def fn(traj):
        traj.steps[2].reroll(3, "subgroup")
        traj.steps[1].reroll()
    use(fn)
    record = _record(_run(_trial()))
    anchors = record["reroll"]["anchors"]
    assert [(a["step"], a["n"], a["credit"]) for a in anchors] == [(1, 1, "member"), (2, 3, "subgroup")]
    assert [[t["step"] for t in a["turns"]] for a in anchors] == [[0], [0, 1]]
    assert record["chain"][0]["reroll_steps"] == [1, 2]


@pytest.mark.parametrize("fn", [
    lambda traj: (traj.steps[1].set_command({"commands": []}), traj.steps[1].reroll()),
    lambda traj: (traj.steps[1].reroll(), traj.steps[1].set_command({"commands": []}), traj.steps[2].reroll()),
    lambda traj: traj.steps[1].reroll(0),
    lambda traj: traj.steps[1].reroll(17),
    lambda traj: traj.steps[1].reroll(1, "group"),
    lambda traj: traj.steps[1].reroll(1, "verified", hint=""),
    lambda traj: traj.steps[1].reroll(1, "verified", hint="x" * 2001),
    lambda traj: traj.steps[1].reroll(1, "verified", hint=3),
    lambda traj: (traj.steps[2].edit_thinking("done", "not yet"), traj.steps[1].reroll()),
    lambda traj: (traj.steps[1].delete(), traj.steps[1].reroll()),
])
def test_misuse_leaves_the_trial_without_a_reroll(use, fn):
    use(fn)
    out = _run(_trial())
    assert _record(out)["chain"][0]["status"] == "error" and "reroll" not in _record(out)
    assert out[0].tokens == _trial()[0].tokens


def test_no_reroll_of_a_trial_split_by_a_summary(use):
    use(lambda traj: traj.steps[0].reroll())
    samples = _trial() + [deepcopy(_trial()[0])]
    samples[1].metadata["leaf"] = {"node_id": 1}
    out = _run(samples)
    assert "summary" in _record(out)["chain"][0]["reason"] and "reroll" not in _record(out)


def test_one_reroll_per_chain(use, monkeypatch):
    fns = {"a": lambda traj: traj.steps[1].reroll(), "b": lambda traj: traj.steps[2].reroll()}
    monkeypatch.setattr(core, "_resolve", lambda name: fns[name])
    monkeypatch.setenv("PRL_TRANSFORM", "a,b")
    record = _record(_run(_trial()))
    assert [e["status"] for e in record["chain"]] == ["ok", "error"]
    assert [a["step"] for a in record["reroll"]["anchors"]] == [1]


# --- the agent -------------------------------------------------------------------------------------------------------

def test_the_agent_replays_the_turns_then_calls_the_policy(tmp_path, monkeypatch):
    Chat = pytest.importorskip("harbor.llms.chat").Chat

    from prl.rollout import reroll_agent
    from prl.rollout.training_terminus import TrainingTerminus2

    request = {"instruction": "TASK", "context_tokens": 1234, "turns": [
        {"step": 0, "reasoning": "look around", "content": '{"commands": []}', "message": "out 1"},
        {"step": 1, "reasoning": "", "content": '{"commands": [1]}', "message": "out 2"}]}
    (tmp_path / "request.json").write_text(json.dumps(request))
    policy = []

    async def call_policy(self, chat, prompt):
        policy.append(prompt)
        return "policy reply"
    monkeypatch.setattr(TrainingTerminus2, "_chat_with_retries", call_policy)
    monkeypatch.setattr(TrainingTerminus2, "__init__", lambda self, *a, **k: None)
    agent = reroll_agent.RerollTerminus2(reroll_file=str(tmp_path / "request.json"))
    chat = Chat(model=None, interleaved_thinking=True)

    first = asyncio.run(agent._chat_with_retries(chat, "TASK on a fresh host"))
    assert (first.content, first.reasoning_content) == ('{"commands": []}', "look around")
    assert asyncio.run(agent._check_proactive_summarization(chat, "TASK", None)) is None  # no summary in the replay
    assert not hasattr(chat, "_tw_last_seq")
    second = asyncio.run(agent._chat_with_retries(chat, "out 1"))
    assert second.reasoning_content is None
    assert chat._tw_last_seq == (4, 1234)  # the trial's own count through the last replayed reply
    assert asyncio.run(agent._chat_with_retries(chat, "out 2 differs")) == "policy reply"
    assert policy == ["out 2 differs"]
    assert chat.messages == [
        {"role": "user", "content": "TASK"},
        {"role": "assistant", "content": '{"commands": []}', "reasoning_content": "look around"},
        {"role": "user", "content": "out 1"}, {"role": "assistant", "content": '{"commands": [1]}'}]
    replay = json.loads((tmp_path / "reroll_replay.json").read_text())
    assert replay == {"turns": 2, "same": 1, "differ": [1], "first_message_same": False,
                      "differ_detail": [{"step": 1, "recorded": "out 2", "executed": "out 2 differs"}]}


def test_the_agent_shows_the_hint_with_the_policys_first_message(tmp_path, monkeypatch):
    """The hint ends the message the policy gets first (its prompt, never trained), once, after the comparison."""
    Chat = pytest.importorskip("harbor.llms.chat").Chat

    from prl.rollout import reroll_agent
    from prl.rollout.training_terminus import TrainingTerminus2

    request = {"instruction": "TASK", "hint": "Hint: look closer.", "turns": [
        {"step": 0, "reasoning": "look around", "content": '{"commands": []}', "message": "out 1"}]}
    (tmp_path / "request.json").write_text(json.dumps(request))
    policy = []

    async def call_policy(self, chat, prompt):
        policy.append(prompt)
        return "policy reply"
    monkeypatch.setattr(TrainingTerminus2, "_chat_with_retries", call_policy)
    monkeypatch.setattr(TrainingTerminus2, "__init__", lambda self, *a, **k: None)
    agent = reroll_agent.RerollTerminus2(reroll_file=str(tmp_path / "request.json"))
    chat = Chat(model=None, interleaved_thinking=True)
    for prompt in ("TASK", "out 1\n", "out 2"):
        asyncio.run(agent._chat_with_retries(chat, prompt))
    assert policy == ["out 1\n\nHint: look closer.", "out 2"]
    replay = json.loads((tmp_path / "reroll_replay.json").read_text())
    assert replay["same"] == 1 and replay["hint_chars"] == len("Hint: look closer.")


def test_the_replay_comparison_ignores_the_host_name_and_blank_lines():
    """In a smoke, a fresh sandbox's host name and tmux's blank lines made 186 of 188 replayed turns 'differ'."""
    pytest.importorskip("harbor")  # prl.rollout.reroll_agent subclasses Harbor's Terminus-2
    from prl.rollout.reroll_agent import _normal

    def screen(host, pad):
        return f"New Terminal Output:\n{pad}root@{host}:/workspace# ls\nfile.txt  \nroot@{host}:/workspace#\n{pad * 3}"
    recorded, replayed = screen("c-6af1dda6-1441eab2-f5ef77a7d6fa", ""), screen("c-6af1ddea-1441eab2-507b6edd6d20", "\n")
    assert _normal(recorded) == _normal(replayed)
    assert _normal(recorded) != _normal(recorded.replace("file.txt", "other.txt"))  # real output still differs
    assert _normal("mail user@example.com: sent") == "mail user@example.com: sent"  # not a prompt (no # or $)


# --- the rollout worker ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Input:
    args: object
    sample: object
    sampling_params: dict
    evaluation: bool = False


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.delenv("PRL_REROLL_TIMEOUT", raising=False)
    monkeypatch.setitem(sys.modules, "miles.utils.types", SimpleNamespace(
        Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED="aborted"))))
    monkeypatch.setitem(sys.modules, "miles.rollout.base_types", SimpleNamespace(GenerateFnOutput=SimpleNamespace))
    args = SimpleNamespace(custom_reward_post_process_path=harbor_training.ADDED_REWARDS)
    pristine = SimpleNamespace(metadata={"instance_id": "task", "prl_training_contract": True}, index=3,
                               group_index=1, rollout_id=None, status="pending", reward=None)
    trial = SimpleNamespace(metadata={"reward": 0.0, "trial_dir": str(tmp_path),
                                      "prl_transform": {"chain": [], "reroll": {"anchors": [{
                                          "step": 2, "n": 1, "credit": "member", "instruction": "TASK", "turns": [],
                                          "trial_dir": str(tmp_path), "token_start": 40, "rewritten_steps": []}]}}},
                            reward=None, status="completed", loss_mask=[1], train_metadata={})
    return Input(args=args, sample=deepcopy(pristine), sampling_params={"sampling_seed": 42}), pristine, trial


def install(monkeypatch, samples, tmp_path=None):
    calls = []

    async def upstream(input):
        calls.append(deepcopy(input))
        if tmp_path is not None:  # as the agent: next to the branch's request
            folder = Path(input.sample.metadata["prl_reroll_file"]).parent
            (folder / "reroll_replay.json").write_text(
                '{"turns": 2, "same": 1, "differ": [1], "differ_detail": [{"step": 1}]}')
        return SimpleNamespace(samples=samples)
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call", SimpleNamespace(generate=upstream))
    return calls


def _reroll(input, pristine, trial):
    return asyncio.run(branching._reroll(input, SimpleNamespace(samples=[trial]), pristine, 7))


def test_a_reroll_joins_the_group_with_its_own_reward(worker, monkeypatch, tmp_path):
    input, pristine, trial = worker
    leaves = [SimpleNamespace(metadata={"reward": 1.0, "trial_dir": "/trials/r1"}, status="completed", reward=None,
                              loss_mask=[0, 1], train_metadata=None) for _ in range(2)]
    calls = install(monkeypatch, leaves, tmp_path)
    out = _reroll(input, pristine, trial)
    assert out.samples == [trial, *leaves]
    call = calls[0]
    folder = tmp_path / "reroll" / "s2_b0"
    assert call.sample.metadata["prl_reroll_file"] == str(folder / "request.json")
    assert call.sample.index == 3 and call.sampling_params["sampling_seed"] == 42 + 3000017
    assert json.loads((folder / "request.json").read_text())["step"] == 2
    [outcome] = trial.metadata["prl_transform"]["reroll"]
    assert outcome["status"] == "ok" and outcome["reward"] == 1.0 and outcome["samples"] == 2
    assert (outcome["step"], outcome["branch"], outcome["credit"]) == (2, 0, "member")
    assert "turns" not in outcome  # the request's text stays in the trial dir
    assert outcome["replay"] == {"turns": 2, "same": 1, "differ": [1]}  # the differing texts stay in the file
    for leaf in leaves:  # the trial's reward on the sample; the branch's as its group member's
        assert leaf.metadata["reward"] == 0.0 and leaf.train_metadata == {
            "group_reward": 1.0, "added_id": "reroll:2:0", "branch": {"anchor": 2, "credit": "member"}}
        assert leaf.metadata["prl_reroll_of"] == str(tmp_path)
    assert "branch_anchors" not in trial.train_metadata  # member credit: the trial keeps its group value
    assert json.loads((folder / "outcome.json").read_text())["status"] == "ok"
    # the branch scored 1.0 against the trial's 0.0: recorded on the outcome and in the trial's transform.json
    assert outcome["beat_trial"] is True and trial.metadata["prl_transform"]["reroll_beaten_steps"] == [2]
    on_disk = json.loads((tmp_path / "transform.json").read_text())
    assert on_disk["reroll"][0]["beat_trial"] is True and on_disk["reroll_beaten_steps"] == [2]


def test_every_branch_runs_with_its_own_seed_and_subgroup_steps_mark_the_trial(worker, monkeypatch, tmp_path):
    input, pristine, trial = worker
    trial.metadata["prl_transform"]["reroll"]["anchors"] = [
        {"step": 1, "n": 1, "credit": "member", "turns": [], "trial_dir": str(tmp_path), "token_start": 20},
        {"step": 2, "n": 2, "credit": "subgroup", "turns": [], "trial_dir": str(tmp_path), "token_start": 40}]
    calls = []

    async def upstream(input):
        calls.append(input)
        return SimpleNamespace(samples=[SimpleNamespace(metadata={"reward": 1.0, "trial_dir": f"/trials/r{len(calls)}"},
                                                        status="completed", reward=None, loss_mask=[0, 1],
                                                        train_metadata=None)])
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call", SimpleNamespace(generate=upstream))
    out = _reroll(input, pristine, trial)
    assert len(calls) == 3 and len(out.samples) == 4
    assert len({c.sampling_params["sampling_seed"] for c in calls}) == 3
    assert sorted(Path(c.sample.metadata["prl_reroll_file"]).parent.name for c in calls) == [
        "s1_b0", "s2_b0", "s2_b1"]
    outcomes = trial.metadata["prl_transform"]["reroll"]
    assert [(o["step"], o["branch"], o["credit"], o["status"]) for o in outcomes] == [
        (1, 0, "member", "ok"), (2, 0, "subgroup", "ok"), (2, 1, "subgroup", "ok")]
    assert sorted(s.train_metadata["added_id"] for s in out.samples[1:]) == ["reroll:1:0", "reroll:2:0", "reroll:2:1"]
    assert trial.train_metadata["branch_anchors"] == [{"step": 2, "start": 40, "credit": "subgroup"}]


def test_a_verified_step_marks_the_trial_with_its_credit(worker, monkeypatch, tmp_path):
    input, pristine, trial = worker
    trial.metadata["prl_transform"]["reroll"]["anchors"][0].update(credit="verified", hint="Hint: x.")
    install(monkeypatch, [SimpleNamespace(metadata={"reward": 1.0, "trial_dir": "/trials/r1"}, status="completed",
                                          reward=None, loss_mask=[0, 1], train_metadata=None)], tmp_path)
    out = _reroll(input, pristine, trial)
    assert trial.train_metadata["branch_anchors"] == [{"step": 2, "start": 40, "credit": "verified"}]
    assert out.samples[1].train_metadata["branch"] == {"anchor": 2, "credit": "verified"}
    assert json.loads((tmp_path / "reroll" / "s2_b0" / "request.json").read_text())["hint"] == "Hint: x."


def test_a_subgroup_step_without_a_successful_branch_leaves_the_trial_unmarked(worker, monkeypatch):
    input, pristine, trial = worker
    trial.metadata["prl_transform"]["reroll"]["anchors"][0]["credit"] = "subgroup"
    install(monkeypatch, [SimpleNamespace(metadata={}, status="aborted", reward=None, loss_mask=[1])])
    out = _reroll(input, pristine, trial)
    assert out.samples == [trial] and "branch_anchors" not in trial.train_metadata


@pytest.mark.parametrize("reject, mismatch, kept", [
    (True, [], True), (True, [{"leaf": 0}], False), (False, [{"leaf": 0}], True)])
def test_a_reroll_with_a_tito_mismatch_is_dropped_when_trials_are(worker, monkeypatch, reject, mismatch, kept):
    """PRL_REJECT_MISMATCH: the trials' check applies to the reroll's samples, without retries."""
    input, pristine, trial = worker
    pristine.metadata["prl_reject_mismatch"] = reject
    checks = {"checked_leaves": 1, "rejected": mismatch}
    leaf = SimpleNamespace(metadata={"reward": 1.0, "trial_dir": "/trials/r1", "tito_session_mismatch": mismatch,
                                     "prl_mismatch_checks": checks},
                           status="completed", reward=None, loss_mask=[0, 1], train_metadata=None)
    install(monkeypatch, [leaf])
    out = _reroll(input, pristine, trial)
    [outcome] = trial.metadata["prl_transform"]["reroll"]
    assert (out.samples == [trial, leaf]) is kept and (outcome["status"] == "ok") is kept
    if not kept:
        assert "TITO mismatch" in outcome["reason"]


@pytest.mark.parametrize("leaves", [
    [SimpleNamespace(metadata={}, status="aborted", reward=None, loss_mask=[1])],
    [SimpleNamespace(metadata={"reward": None}, status="completed", reward=None, loss_mask=[1])],
    [SimpleNamespace(metadata={"reward": 1.0}, status="completed", reward=None, loss_mask=[0])],
])
def test_a_failed_reroll_leaves_the_trial_alone(worker, monkeypatch, leaves):
    input, pristine, trial = worker
    install(monkeypatch, leaves)
    out = _reroll(input, pristine, trial)
    assert out.samples == [trial] and trial.metadata["prl_transform"]["reroll"][0]["status"] == "failed"


@pytest.mark.parametrize("aborts, status", [(1, "ok"), (2, "failed")])
def test_an_aborted_branch_runs_again_with_another_seed(worker, monkeypatch, aborts, status):
    """An abort is Harbor's or the session collect's failure, not a score: it runs again, as an
    aborted trial does (PRL_ABORT_MAX_RETRIES)."""
    input, pristine, trial = worker
    monkeypatch.setenv("PRL_ABORT_MAX_RETRIES", "1")
    calls = []

    async def upstream(input):
        calls.append(input)
        if len(calls) <= aborts:
            return SimpleNamespace(samples=[SimpleNamespace(metadata={}, status="aborted", reward=None, loss_mask=[1])])
        return SimpleNamespace(samples=[SimpleNamespace(metadata={"reward": 1.0, "trial_dir": "/trials/r2"},
                                                        status="completed", reward=None, loss_mask=[0, 1],
                                                        train_metadata=None)])
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call", SimpleNamespace(generate=upstream))
    out = _reroll(input, pristine, trial)
    [outcome] = trial.metadata["prl_transform"]["reroll"]
    assert len(calls) == 2 and outcome["attempts"] == 2 and outcome["status"] == status
    assert calls[1].sampling_params["sampling_seed"] == calls[0].sampling_params["sampling_seed"] + 2000003
    if status == "ok":
        assert outcome["trial_dir"] == "/trials/r2" and len(out.samples) == 2
    else:
        assert "aborted 2 times" in outcome["reason"] and out.samples == [trial]


def test_a_reroll_past_its_deadline_is_dropped(worker, monkeypatch):
    input, pristine, trial = worker
    monkeypatch.setenv("PRL_REROLL_TIMEOUT", "0.01")

    async def slow(input):
        await asyncio.sleep(1)
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call", SimpleNamespace(generate=slow))
    out = _reroll(input, pristine, trial)
    [outcome] = trial.metadata["prl_transform"]["reroll"]
    assert out.samples == [trial] and "PRL_REROLL_TIMEOUT" in outcome["reason"]


def test_a_reroll_needs_the_group_reward_postprocess(worker, monkeypatch):
    input, pristine, trial = worker
    install(monkeypatch, [])
    with pytest.raises(ValueError, match="custom-reward-post-process-path"):
        _reroll(Input(args=SimpleNamespace(), sample=input.sample, sampling_params={}), pristine, trial)


def test_no_request_no_reroll(worker, monkeypatch):
    input, pristine, trial = worker
    calls = install(monkeypatch, [])
    del trial.metadata["prl_transform"]["reroll"]
    result = SimpleNamespace(samples=[trial])
    assert asyncio.run(branching._reroll(input, result, pristine, 7)) is result and not calls


# --- metrics ---------------------------------------------------------------------------------------------------------

def _metric_sample(trial_dir, reward, **md):
    return SimpleNamespace(tokens=[1, 2, 3], response_length=2, loss_mask=[1, 1], rollout_log_probs=None,
                           get_reward_value=lambda args, r=reward: r, metadata={"trial_dir": trial_dir, **md})


def test_reroll_metrics():
    record = {"chain": [{"transform": "t", "status": "ok"}], "trained_tokens_before": 2, "trained_tokens_after": 2,
              "changed": True}
    branch = {"status": "ok", "reward": 1.0, "seconds": 30.0, "credit": "subgroup"}
    ok = {**record, "reroll": [branch, {"status": "ok", "reward": 0.0, "seconds": 20.0, "credit": "member"}]}
    failed = {**record, "reroll": [{"status": "failed", "seconds": 10.0, "credit": "member"}]}
    samples = [_metric_sample("a", 0.0, prl_transform=ok), _metric_sample("b", 1.0, prl_transform=failed),
               _metric_sample("c", 1.0, prl_transform=record),
               _metric_sample("r", 0.0, prl_reroll=branch, agent_metrics={"n_episodes": 99})]
    out = compute(SimpleNamespace(), samples)
    assert (out["rollout/reroll/requested_trials"], out["rollout/reroll/branches"], out["rollout/reroll/ok_branches"],
            out["rollout/reroll/failed_branches"], out["rollout/reroll/subgroup_branches"]) == (2, 3, 2, 1, 1)
    assert out["rollout/reroll/reward_mean"] == 0.5 and out["rollout/reroll/source_reward_mean"] == 0.0
    assert out["rollout/reroll/seconds_mean"] == 20.0
    assert out["rollout/transform/ok_trials"] == 3 and "rollout/trial_turns/max" not in out  # the reroll is no trial
    assert out["rollout/raw_reward_min"] == 0.0 and abs(out["rollout/raw_reward_std"] - 0.4714) < 1e-3


def test_reroll_metrics_count_the_confirmed_verified_steps():
    record = {"chain": [{"transform": "t", "status": "ok"}], "trained_tokens_before": 2, "trained_tokens_after": 2,
              "changed": True}

    def branch(step, status, reward=None):
        return {"step": step, "status": status, "reward": reward, "seconds": 1.0, "credit": "verified"}
    confirmed = {**record, "reroll": [branch(3, "ok", 1.0), branch(3, "ok", 0.0)]}
    unconfirmed = {**record, "reroll": [branch(2, "ok", 0.0), branch(2, "failed")]}
    out = compute(SimpleNamespace(), [_metric_sample("a", 0.0, prl_transform=confirmed),
                                      _metric_sample("b", 0.0, prl_transform=unconfirmed)])
    assert (out["rollout/reroll/verified_steps"], out["rollout/reroll/confirmed_steps"]) == (2, 1)
    assert "rollout/reroll/verified_steps" not in compute(SimpleNamespace(), [
        _metric_sample("c", 0.0, prl_transform={**record, "reroll": [{**branch(1, "ok", 1.0), "credit": "subgroup"}]})])


def test_a_reroll_after_an_earlier_function_masked_a_turn_replays_it(monkeypatch):
    """The chain's second function sees the masked turn as a step: the replay has every turn before the anchor, and
    the anchor's first token is where the turn really starts (a smoke had replayed one turn of three)."""
    for key in ("PRL_TRANSFORM_ARGS", "PRL_TRANSFORM_MAX_FAILURES", "PRL_TRANSFORM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(core, "_failures", 0)

    def mask(traj):
        traj.steps[1].mask()

    def fork(traj):
        traj.record["masked"] = [s.masked for s in traj.steps]
        traj.steps[2].reroll(2, "verified")
    fns = {"m": mask, "f": fork}
    monkeypatch.setattr(core, "_resolve", lambda name: fns[name])
    monkeypatch.setenv("PRL_TRANSFORM", "m+f")
    alone = _record(_run(_trial()))  # reference: the same reroll without the mask
    monkeypatch.setenv("PRL_TRANSFORM", "f")
    reference = _record(_run(_trial()))["reroll"]["anchors"][0]
    assert alone["chain"][1]["record"]["masked"] == [False, True, False]
    [request] = alone["reroll"]["anchors"]
    assert [t["step"] for t in request["turns"]] == [0, 1] and request["step"] == 2
    assert request["token_start"] == reference["token_start"]
