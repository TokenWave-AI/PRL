"""Protect per-trial evaluation weighting when summaries create TITO branches."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from prl.rollout.eval_sample_picker import latest_eval_leaf


def _sample(node_id: int, reward: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(metadata={"leaf": {"node_id": node_id}}, tokens=[node_id], reward=reward)


def test_summary_branches_do_not_reweight_trial(monkeypatch):
    monkeypatch.setenv("MODE", "rollout")
    # A successful trial with three branches must count once, like a failed
    # trial without summaries: mean reward is 1/2, not 3/4.
    success = [_sample(8, 1.0), _sample(1, 1.0), _sample(4, 1.0)]
    failed = [_sample(0)]
    before = deepcopy(success)
    picked = latest_eval_leaf(success, {}) + latest_eval_leaf(failed, {})
    assert len(picked) == 2
    assert sum(sample.reward for sample in picked) / len(picked) == 0.5
    assert picked[0] is success[0]
    assert success == before


def test_empty_session_stays_empty(monkeypatch):
    monkeypatch.setenv("MODE", "rollout")
    assert latest_eval_leaf([], {}) == []


@pytest.mark.parametrize("mode", [None, "train"])
def test_training_or_missing_mode_is_rejected(monkeypatch, mode):
    monkeypatch.delenv("MODE", raising=False)
    if mode is not None:
        monkeypatch.setenv("MODE", mode)
    with pytest.raises(ValueError, match="requires MODE=rollout"):
        latest_eval_leaf([_sample(0)], {})


def test_missing_leaf_metadata_fails_loudly(monkeypatch):
    monkeypatch.setenv("MODE", "rollout")
    with pytest.raises(KeyError):
        latest_eval_leaf([SimpleNamespace(metadata={})], {})
