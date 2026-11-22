"""An aborted training trial runs again on its own (prl/rollout/harbor_training.py ``_retry_aborted``), so Miles
does not discard and refill its whole group; no GPU, sandbox or Miles needed (fake generator, fake Sample status)."""
from __future__ import annotations

import asyncio
import sys
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from prl.rollout.harbor_training import generate
from prl.rollout.rollout_metrics import compute


@dataclass(frozen=True)
class Input:
    args: object
    sample: object
    sampling_params: dict
    evaluation: bool = False


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("PRL_REJECT_MISMATCH", "0")
    monkeypatch.delenv("PRL_ABORT_MAX_RETRIES", raising=False)  # default 2
    monkeypatch.delenv("PRL_TRANSFORM", raising=False)
    monkeypatch.setitem(sys.modules, "miles.utils.types", SimpleNamespace(
        Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED="aborted"))))
    monkeypatch.setitem(sys.modules, "miles.rollout.base_types", SimpleNamespace(GenerateFnOutput=SimpleNamespace))
    sample = SimpleNamespace(metadata={"instance_id": "task"}, rollout_id=17, index=17, prompt="same task",
                             reward=None, status="pending")
    return Input(args=SimpleNamespace(use_session_server="v2"), sample=sample, sampling_params={"sampling_seed": 42})


def done(reward=1):
    return SimpleNamespace(metadata={}, status="completed", reward=reward, rollout_id=None)


def aborted(**metadata):
    return SimpleNamespace(metadata=dict(metadata), status="aborted", reward=None, rollout_id=None)


def install(monkeypatch, outputs):
    calls = []

    async def upstream(input):
        calls.append(deepcopy(input))
        input.sample.metadata["upstream_mutation"] = True  # Miles' generator writes into the sample
        return SimpleNamespace(samples=outputs[len(calls) - 1])
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call", SimpleNamespace(generate=upstream))
    return calls


def test_aborted_trial_is_run_again_alone_with_a_clean_sample_and_new_seed(setup, monkeypatch):
    ok = done(0)
    calls = install(monkeypatch, [[aborted()], [ok]])
    result = asyncio.run(generate(setup))
    assert result.samples == [ok]
    assert ok.reward == 0 and ok.rollout_id == 17 and ok.metadata["prl_abort_retries"] == 1
    assert [c.sampling_params["sampling_seed"] for c in calls] == [42, 42 + 2000003]
    assert calls[1].sample.prompt == "same task" and calls[1].sample.index == 17
    assert "upstream_mutation" not in calls[1].sample.metadata  # the retry starts from the sample before attempt 1


def test_first_attempt_is_unchanged_and_records_zero_retries(setup, monkeypatch):
    ok = done()
    calls = install(monkeypatch, [[ok]])
    assert asyncio.run(generate(setup)).samples == [ok]
    assert len(calls) == 1 and calls[0].sampling_params["sampling_seed"] == 42
    assert ok.metadata["prl_abort_retries"] == 0


def test_every_attempt_aborted_reaches_miles_as_an_abort(setup, monkeypatch):
    calls = install(monkeypatch, [[aborted()] for _ in range(3)])
    result = asyncio.run(generate(setup))
    assert len(calls) == 3 and [s.status for s in result.samples] == ["aborted"]


def test_zero_retries_is_the_old_group_refill(setup, monkeypatch):
    monkeypatch.setenv("PRL_ABORT_MAX_RETRIES", "0")
    ab = aborted()
    calls = install(monkeypatch, [[ab]])
    assert asyncio.run(generate(setup)).samples == [ab]
    assert len(calls) == 1 and "prl_abort_retries" not in ab.metadata


def install_eval(monkeypatch, outputs, sandbox):
    """As install, but each attempt also reports what made it abort, as harbor_training.run does."""
    from prl.rollout import harbor_training

    calls = []

    async def upstream(input):
        calls.append(deepcopy(input))
        note = harbor_training._failure.get()
        assert note == {}  # a fresh note per attempt
        note["sandbox"] = sandbox[len(calls) - 1]
        return SimpleNamespace(samples=outputs[len(calls) - 1])
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call", SimpleNamespace(generate=upstream))
    return calls


def test_evaluation_runs_a_sandbox_failure_again_when_asked(setup, monkeypatch):
    setup = Input(setup.args, setup.sample, setup.sampling_params, evaluation=True)
    monkeypatch.setenv("PRL_EVAL_SANDBOX_RETRIES", "2")
    ok = done(0)
    calls = install_eval(monkeypatch, [[aborted()], [aborted()], [ok]], sandbox=[True, True, False])
    assert asyncio.run(generate(setup)).samples == [ok] and len(calls) == 3
    assert [c.sampling_params["sampling_seed"] for c in calls] == [42, 42 + 2000003, 42 + 2 * 2000003]


def test_evaluation_scores_any_other_abort_zero(setup, monkeypatch):
    setup = Input(setup.args, setup.sample, setup.sampling_params, evaluation=True)
    monkeypatch.setenv("PRL_EVAL_SANDBOX_RETRIES", "2")
    ab = aborted()
    calls = install_eval(monkeypatch, [[ab], [done()]], sandbox=[False, False])  # e.g. the model closed its shell
    assert asyncio.run(generate(setup)).samples == [ab] and len(calls) == 1


def test_evaluation_scores_a_sandbox_failure_zero_by_default(setup, monkeypatch):
    setup = Input(setup.args, setup.sample, setup.sampling_params, evaluation=True)
    monkeypatch.delenv("PRL_EVAL_SANDBOX_RETRIES", raising=False)
    ab = aborted()
    calls = install_eval(monkeypatch, [[ab], [done()]], sandbox=[True, False])
    assert asyncio.run(generate(setup)).samples == [ab] and len(calls) == 1


@pytest.mark.parametrize("message, sandbox", [
    ("Harbor RateLimitException: 429: Rate limit exceeded - account resource quota exhausted; inspect trial x", True),
    ("Harbor SandboxException: Code.INTERNAL: fork/exec /bin/bash: invalid argument; inspect trial x", True),
    ("Harbor Exception: Command ended without an end event; inspect trial x", True),
    ("Harbor ReadTimeout: timed out; inspect trial x", True),  # the e2b SDK's requests to the service
    ("Harbor RemoteProtocolError: Request failed: error sending request for url; inspect trial x", True),
    ("Harbor TimeoutException: Request timed out: the stream didn't open within 'request_timeout'; inspect trial x",
     True),
    ("Harbor AgentTimeoutError: Agent execution timed out after 900.0 seconds; inspect trial x", False),
    ("Harbor RuntimeError: failed to send non-blocking keys: command=\"tmux send-keys\"; inspect trial x", False),
    ("Harbor RewardFileNotFoundError: No reward file found; inspect trial x", False),
])
def test_run_tells_a_sandbox_failure_from_others(monkeypatch, message, sandbox):
    from prl.rollout import harbor_training

    async def fail(*args, **kwargs):
        raise RuntimeError(message)
    monkeypatch.setitem(sys.modules, "prl.rollout.harbor_agent", SimpleNamespace(run=fail))
    monkeypatch.setenv("PRL_RATE_LIMIT_WAIT", "0")

    async def one():
        note = {}
        harbor_training._failure.set(note)
        out = await harbor_training.run(metadata={"prl_evaluation": True})
        return out, note
    out, note = asyncio.run(one())
    assert out["prl_aborted"] is True and out["reward"] is None and note == {"sandbox": sandbox}


def test_mismatch_exhaustion_is_not_retried_again(setup, monkeypatch):
    monkeypatch.setenv("PRL_REJECT_MISMATCH", "1")
    monkeypatch.setenv("PRL_MISMATCH_MAX_RETRIES", "0")
    monkeypatch.setenv("ASYNC", "0")
    bad = SimpleNamespace(metadata={"tito_session_mismatch": [{"type": "assistant_text"}],
                                    "prl_mismatch_checks": {"checked_leaves": 1, "rejected": [1]}},
                          status="completed", reward=1, rollout_id=None, tokens=[1], response_length=1, loss_mask=[1])
    calls = install(monkeypatch, [[bad], [done()]])
    result = asyncio.run(generate(setup))
    assert len(calls) == 1 and result.samples[0].metadata["prl_mismatch_exhausted"]


def test_negative_retries_rejected(setup, monkeypatch):
    monkeypatch.setenv("PRL_ABORT_MAX_RETRIES", "-1")
    install(monkeypatch, [[done()]])
    with pytest.raises(ValueError, match="PRL_ABORT_MAX_RETRIES"):
        asyncio.run(generate(setup))


def test_metric_counts_trials_kept_after_a_retry():
    def s(trial, retries):
        return SimpleNamespace(metadata={"trial_dir": trial, "prl_abort_retries": retries}, tokens=None,
                               response_length=0, rollout_log_probs=None, loss_mask=[],
                               get_reward_value=lambda args: 1.0)
    out = compute(None, [s("a", 0), s("b", 1), s("b", 1), s("c", 2)])
    assert out["rollout/abort_retried_trials"] == 2  # b (two samples, one trial) and c
    assert "rollout/abort_retried_trials" not in compute(None, [SimpleNamespace(
        metadata={}, tokens=None, response_length=0, rollout_log_probs=None, loss_mask=[],
        get_reward_value=lambda args: 1.0)])
