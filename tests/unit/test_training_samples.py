"""Exercise real Miles shared-prefix masking with synthetic compact trajectories."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from prl.rollout.harbor_training import generate, run
from prl.rollout.training_samples import mask_auxiliary_completions, pick_samples, postprocess_samples


@pytest.fixture
def upstream_postprocessor(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "build/miles/miles/rollout/session/v2/postprocessor_hub/default_postprocess.py"
    if not path.exists():
        pytest.skip("Generate build/miles with scripts/setup/apply_patches.sh")
    monkeypatch.setitem(sys.modules, "miles.utils.types", SimpleNamespace(Sample=SimpleNamespace))
    name = "miles.rollout.session.v2.postprocessor_hub.default_postprocess"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)


def _sample(leaf, path, length, mask):
    return SimpleNamespace(tokens=list(range(length)), response_length=length - 2, loss_mask=mask,
                           metadata={"leaf": {"node_id": leaf, "path_node_ids": path}}, reward=None)


def _node(node_id, span, summary=False):
    return {"id": node_id, "completion_span": span,
            "turn_args": {"user": "prl/summary-v1" if summary else "prl/action-v1"}}


def test_summary_before_and_after_actions_are_trained_once(upstream_postprocessor):
    # Two summary-related leaves share the first action. A question-only root
    # must disappear; a new main-action root after compact must remain.
    before = _sample(1, [0, 1], 10, [1, 1, 1, 0, 1, 1, 1, 1])
    questions = _sample(2, [2], 5, [1, 1, 1])
    answers = _sample(3, [0, 1, 3], 13, [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
    after = _sample(4, [4], 6, [0, 1, 1, 1])
    session = {"agent": {"reward": 0.75, "prl_evaluation": False}, "tree": {"nodes": [
        _node(0, [2, 5]), _node(1, [6, 10], True), _node(2, [2, 5], True),
        _node(3, [11, 13], True), _node(4, [3, 6]),
    ]}}
    kept = postprocess_samples([before, questions, answers, after], session)
    assert kept == [before, after]
    assert before.loss_mask == [1, 1, 1, 0, 0, 0, 0, 0]
    assert after.loss_mask == [0, 1, 1, 1]
    assert sum(map(sum, (s.loss_mask for s in kept))) == 6
    assert all(s.reward == 0.75 for s in kept)


def test_clamp_aux_span_to_truncated_response():
    sample = _sample(1, [0, 1], 6, [1, 1, 1, 1])
    metadata = {"tree": {"nodes": [_node(0, [2, 4]), _node(1, [4, 20], True)]}}
    assert mask_auxiliary_completions([sample], metadata) == [sample]
    assert sample.loss_mask == [1, 1, 0, 0]


def test_unknown_tags_fail_closed():
    sample = _sample(0, [0], 4, [1, 1])
    with pytest.raises(ValueError, match="training call tag"):
        mask_auxiliary_completions([sample], {"tree": {"nodes": [
            {"id": 0, "completion_span": [2, 4], "turn_args": {}},
        ]}})


def test_periodic_eval_uses_request_flag_not_process_mode(monkeypatch, upstream_postprocessor):
    monkeypatch.setenv("MODE", "train")
    samples = [_sample(1, [1], 4, [1, 1]), _sample(3, [3], 4, [1, 1])]
    metadata = {"agent": {"prl_training_contract": True, "prl_evaluation": True, "reward": 0},
                "tree": {"nodes": [_node(1, [2, 4], True), _node(3, [2, 4])]}}
    picked = pick_samples(samples, metadata)
    assert picked == [samples[1]]
    assert postprocess_samples(picked, metadata)[0].reward == 0


@pytest.mark.parametrize("agent", [None, {}, {"prl_training_contract": True, "prl_aborted": True}])
def test_infra_or_missing_metadata_collects_no_trainable_samples(agent):
    assert pick_samples([_sample(0, [0], 4, [1, 1])], {"agent": agent}) == []


@pytest.mark.parametrize("case", ["exception", "unscored", "real_zero"])
def test_agent_error_is_aborted_but_verifier_zero_is_kept(monkeypatch, case):
    async def fake_harbor(**kwargs):
        if case == "exception":
            raise RuntimeError("sandbox unavailable")
        return {"reward": 0.0, "eval_report": {"reward": 0.0} if case == "real_zero" else {},
                "exit_status": "Submitted"}
    monkeypatch.setitem(sys.modules, "prl.rollout.harbor_agent", SimpleNamespace(run=fake_harbor))
    result = asyncio.run(run(metadata={"prl_evaluation": False}))
    assert result["prl_aborted"] == (case != "real_zero")
    assert result["reward"] == (0.0 if case == "real_zero" else None)


def test_compact_siblings_keep_one_rollout_id(monkeypatch):
    samples = [SimpleNamespace(rollout_id=None, status="completed", metadata={}) for _ in range(2)]
    async def fake_generate(input):
        assert input.sample.metadata["prl_evaluation"] is False
        return SimpleNamespace(samples=samples)
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call",
                        SimpleNamespace(generate=fake_generate))
    # the abort retry (harbor_training._retry_aborted) reads the samples' status
    monkeypatch.setitem(sys.modules, "miles.utils.types", SimpleNamespace(
        Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED="aborted"))))
    sample = SimpleNamespace(metadata={}, rollout_id=None, index=7)
    input = SimpleNamespace(args=SimpleNamespace(use_session_server="v2"), sample=sample, evaluation=False)
    assert asyncio.run(generate(input)).samples == samples
    assert [s.rollout_id for s in samples] == [7, 7]
