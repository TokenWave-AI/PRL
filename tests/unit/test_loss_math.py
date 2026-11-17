"""Miles' loss math as our arms use it (KL_LOSS_TYPE=k3, --calculate-per-token-loss), against slow simple references.
Needs torch and Miles (the Miles image: CHAIN=unit); a venv without them skips these tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2] / "build/miles"


def _miles(monkeypatch, module: str):
    if not (ROOT / "miles").exists():
        pytest.skip("Generate build/miles with scripts/setup/apply_patches.sh")
    monkeypatch.syspath_prepend(str(ROOT))
    try:
        return __import__(module, fromlist=["_"])
    except Exception as e:  # noqa: BLE001 - the host venv lacks Miles' dependencies (any of them)
        pytest.skip(f"Miles not importable: {e}")


def test_k3_is_kl_in_expectation_and_its_gradient_is_analytic(monkeypatch):
    """configs/arms/common_ota.env: --use-kl-loss --kl-loss-type k3 against the reference model."""
    math_utils = _miles(monkeypatch, "miles.backends.training_utils.loss.hub.math_utils")
    torch.manual_seed(0)
    logits, base_logits = torch.randn(50, dtype=torch.float64), torch.randn(50, dtype=torch.float64)
    p, q = torch.log_softmax(logits, -1), torch.log_softmax(base_logits, -1)
    k3 = math_utils.compute_approx_kl(p, q, "k3")
    # E_{x ~ pi}[k3(x)] = KL(pi || ref), exactly (sum over the whole vocabulary)
    assert torch.allclose((p.exp() * k3).sum().double(), (p.exp() * (p - q)).sum(), atol=1e-6)
    # gradient per sampled token: d/d log pi [exp(q - log pi) - 1 - (q - log pi)] = 1 - exp(q - log pi)
    lp = p.clone().float().requires_grad_(True)
    math_utils.compute_approx_kl(lp, q.float(), "k3").sum().backward()
    assert torch.allclose(lp.grad, (1 - (q - p).exp()).float(), atol=1e-5)
    # at the reference policy: value and gradient 0 (step-0 invariant, kl_loss 0 in the smoke acceptance)
    same = q.clone().float().requires_grad_(True)
    value = math_utils.compute_approx_kl(same, q.float(), "k3")
    value.sum().backward()
    assert value.abs().max() == 0 and same.grad.abs().max() == 0


def test_the_per_token_loss_does_not_depend_on_how_the_batch_is_split(monkeypatch):
    """--calculate-per-token-loss: each micro-batch returns the masked token sum and its token count (objective.py),
    Megatron adds both over micro-batches and data-parallel ranks and divides once. Every split of a batch then gives
    the global masked token mean. (A sample with no trained token counts 1 in the denominator: per sample, so still
    split-invariant.)"""
    cp = _miles(monkeypatch, "miles.backends.training_utils.data.context_parallel")
    monkeypatch.setattr(cp, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1, rank=0)))
    torch.manual_seed(1)
    lengths = [5, 9, 3, 7, 4, 6]
    masks = [(torch.rand(n) > 0.4).float() for n in lengths]
    masks[2] = torch.zeros(lengths[2])  # e.g. every turn of a trial masked by a transform
    losses = [torch.randn(n, dtype=torch.float64) for n in lengths]

    def partial(indices):
        resp = [lengths[i] for i in indices]
        sum_of_token = cp.get_sum_of_sample_mean(resp, resp, [masks[i] for i in indices], True, "thd")
        total = sum_of_token(torch.cat([losses[i] for i in indices]))
        return total, sum(torch.clamp_min(masks[i].sum(), 1) for i in indices)

    def loss(split):
        parts = [partial(indices) for indices in split]
        return sum(t for t, _ in parts) / sum(n for _, n in parts)

    reference = sum((l * m).sum() for l, m in zip(losses, masks)) / sum(max(m.sum(), 1) for m in masks)
    splits = ([[0, 1, 2, 3, 4, 5]], [[0], [1], [2], [3], [4], [5]], [[0, 3], [1, 2, 4], [5]], [[5, 1], [0, 2, 3, 4]])
    for split in splits:
        assert torch.allclose(loss(split), reference), split
