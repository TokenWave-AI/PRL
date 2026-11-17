"""The smoke acceptance (scripts/checks/grpo_acceptance.py check_batch) on a TITO mismatch: the synchronous smoke
runs a mismatched trial again (PRL_REJECT_MISMATCH=1), so a batch must hold none; the async smoke cannot
(prl/rollout/harbor_training.py allows the refill only in synchronous training), so its
mismatched trials train and are counted."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
import grpo_acceptance


def _batch(tmp_path: Path, reject: bool, mismatched: int = 0) -> list[dict]:
    """8 groups x 8 one-leaf trials, as the smoke's rollout dump holds them: prompt [1, 2], response [3, 4] generated
    by one model call of weight version 1; the first ``mismatched`` trials carry a TITO mismatch."""
    samples = []
    for n in range(64):
        trial = tmp_path / f"trial_{n}"
        trial.mkdir()
        (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": float(n % 2)}}}))
        meta = {"trial_dir": str(trial), "accumulated_token_ids": [1, 2, 3, 4], "prl_reject_mismatch": reject,
                "tito_session_mismatch": [{"type": "assistant_text"}] if n < mismatched else []}
        if reject:
            meta |= {"prl_mismatch_checks": {"checked_leaves": 1, "rejected": []}, "prl_mismatch_retries": 0}
        samples.append({"metadata": meta, "status": "completed", "reward": float(n % 2), "tokens": [1, 2, 3, 4],
                        "loss_mask": [1, 1], "response_length": 2, "group_index": n // 8, "rollout_id": n,
                        "weight_versions": [[{"abs_start": 2, "abs_end": 4, "version": "1"}]]})
    return samples


def _check(monkeypatch, samples: list[dict], **kwargs) -> dict:
    monkeypatch.setattr(grpo_acceptance, "load_samples", lambda run, batch: samples)
    return grpo_acceptance.check_batch(Path("unused"), 0, **kwargs)


def test_synchronous_smoke_holds_no_mismatch(tmp_path, monkeypatch):
    assert _check(monkeypatch, _batch(tmp_path, reject=True))["accepted_mismatches"] == 0


def test_async_smoke_trains_and_counts_mismatches(tmp_path, monkeypatch):
    samples = _batch(tmp_path, reject=False, mismatched=3)
    result = _check(monkeypatch, samples, staleness=2, reject=False)
    assert result["accepted_mismatches"] == 3 and result["retried_rollouts"] == 0
    with pytest.raises(AssertionError):  # judged as a synchronous smoke, the same batch fails
        _check(monkeypatch, samples, staleness=2)


def test_a_mismatch_left_in_a_synchronous_batch_fails(tmp_path, monkeypatch):
    samples = _batch(tmp_path, reject=True)
    samples[5]["metadata"]["tito_session_mismatch"] = [{"type": "assistant_text"}]
    with pytest.raises(AssertionError):
        _check(monkeypatch, samples)
