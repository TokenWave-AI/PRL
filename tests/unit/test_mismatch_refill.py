"""Validate trial-level rejection/refill without a GPU or cloud sandbox.

Fake generation represents separate TITO sessions. The real adapter and picker
must preserve grouping, exclude summary-only mismatches, and keep eval untouched.
"""
from __future__ import annotations

import asyncio
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from prl.rollout.harbor_training import generate
from prl.rollout.training_samples import pick_samples


@dataclass(frozen=True)
class Input:
    args: object
    sample: object
    sampling_params: dict
    evaluation: bool = False


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("PRL_REJECT_MISMATCH", "1")
    monkeypatch.setenv("PRL_MISMATCH_MAX_RETRIES", "2")
    monkeypatch.setenv("ASYNC", "0")
    monkeypatch.setitem(sys.modules, "miles.utils.types", SimpleNamespace(
        Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED="aborted"))))
    monkeypatch.setitem(sys.modules, "miles.rollout.base_types", SimpleNamespace(GenerateFnOutput=SimpleNamespace))
    sample = SimpleNamespace(metadata={"instance_id": "task"}, rollout_id=17, index=17,
                             prompt="same task", reward=None, status="pending")
    return Input(args=SimpleNamespace(use_session_server="v2"), sample=sample, sampling_params={"sampling_seed": 42})


def leaf(bad=False, reward=0):
    mismatch = [{"type": "assistant_text"}] if bad else []
    return SimpleNamespace(metadata={"tito_session_mismatch": mismatch,
                           "prl_mismatch_checks": {"checked_leaves": 2, "rejected": mismatch}},
                           status="completed", reward=reward, rollout_id=None,
                           tokens=[1, 2], response_length=1, loss_mask=[1])


def install(monkeypatch, outputs):
    calls = []
    async def upstream(input):
        calls.append(deepcopy(input))
        input.sample.metadata["upstream_mutation"] = True
        return SimpleNamespace(samples=outputs[len(calls) - 1])
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call", SimpleNamespace(generate=upstream))
    return calls


def test_same_prompt_retries_replace_whole_trial_preserve_zero_reward(setup, monkeypatch, tmp_path):
    bad, good1, good2 = leaf(True, 1), leaf(), leaf()
    bad.metadata["trial_dir"] = str(tmp_path)
    calls = install(monkeypatch, [[bad], [good1, good2]])
    result = asyncio.run(generate(setup))
    assert result.samples == [good1, good2]
    assert all(s.reward == 0 and s.rollout_id == 17 for s in result.samples)
    assert all(s.metadata["prl_mismatch_retries"] == 1 for s in result.samples)
    assert all(c.sample.prompt == "same task" and c.sample.index == 17 for c in calls)
    assert [c.sampling_params["sampling_seed"] for c in calls] == [42, 1000045]
    assert "upstream_mutation" not in calls[1].sample.metadata
    assert "upstream_mutation" not in setup.sample.metadata
    evidence = json.loads((tmp_path / "training_mismatch_rejection.json").read_text())
    assert evidence["samples"][0]["reward"] == 1


def test_exhaustion_returns_one_abort_with_no_reward(setup, monkeypatch):
    calls = install(monkeypatch, [[leaf(True)] for _ in range(3)])
    result = asyncio.run(generate(setup))
    assert len(calls) == 3 and len(result.samples) == 1
    sample = result.samples[0]
    assert sample.status == "aborted" and sample.reward is None and sample.rollout_id == 17
    assert sample.metadata["prl_mismatch_exhausted"]


@pytest.mark.parametrize("mode", ["eval", "disabled"])
def test_eval_and_rollback_do_not_filter(setup, monkeypatch, mode):
    if mode == "eval":
        setup = Input(setup.args, setup.sample, setup.sampling_params, evaluation=True)
    else:
        monkeypatch.setenv("PRL_REJECT_MISMATCH", "0")
    bad = leaf(True)
    calls = install(monkeypatch, [[bad]])
    assert asyncio.run(generate(setup)).samples == [bad]
    assert len(calls) == 1 and not calls[0].sample.metadata["prl_reject_mismatch"]


def test_infra_abort_not_retried_by_mismatch_refill(setup, monkeypatch):
    # An infra abort is not a mismatch; with PRL_ABORT_MAX_RETRIES=0 it goes straight to Miles' group refill
    # (the abort retry itself: tests/unit/test_abort_retry.py)
    monkeypatch.setenv("PRL_ABORT_MAX_RETRIES", "0")
    aborted = leaf(); aborted.status = "aborted"; aborted.reward = None
    calls = install(monkeypatch, [[aborted]])
    assert asyncio.run(generate(setup)).samples == [aborted]
    assert len(calls) == 1


@pytest.mark.parametrize("missing", ["prl_mismatch_checks", "tito_session_mismatch"])
def test_missing_check_cannot_be_accepted(setup, monkeypatch, missing):
    monkeypatch.setenv("PRL_MISMATCH_MAX_RETRIES", "0")
    sample = leaf(); sample.metadata.pop(missing)
    install(monkeypatch, [[sample]])
    assert asyncio.run(generate(setup)).samples[0].status == "aborted"


def test_picker_records_auxiliary_mismatch_before_it_is_dropped(monkeypatch):
    main, aux = leaf(), leaf(True)
    main.metadata["leaf"] = {"node_id": 0}
    aux.metadata["leaf"] = {"node_id": 1}
    monkeypatch.setitem(sys.modules, "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
                        SimpleNamespace(drop_same_prompt_retries=lambda samples, metadata: samples[:1]))
    session = {"agent": {"prl_training_contract": True, "prl_evaluation": False,
                         "prl_reject_mismatch": True}}
    assert pick_samples([main, aux], session) == [main]
    checks = main.metadata["prl_mismatch_checks"]
    assert checks["checked_leaves"] == 2 and checks["rejected"][0]["leaf"]["node_id"] == 1


def test_async_is_rejected(setup, monkeypatch):
    install(monkeypatch, [])
    monkeypatch.setenv("ASYNC", "1")
    with pytest.raises(ValueError, match="synchronous"):
        asyncio.run(generate(setup))
