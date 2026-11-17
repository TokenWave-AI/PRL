import pytest

torch = pytest.importorskip("torch")  # the Miles image (CHAIN=unit)

from prl.algo import fused_logprob as fl
from prl.algo.fused_logprob import LinearLogProb


def _naive(h, w, y):
    logits = (h.reshape(-1, h.size(-1)) @ w.t()).float()
    return (torch.logsumexp(logits, -1) - logits.gather(1, y.reshape(-1, 1)).squeeze(1)).view(y.shape)


def test_matches_naive_values_and_grads(monkeypatch):
    monkeypatch.setattr(fl, "VOCAB_BLOCK", 16)  # the weight gradient in 4 vocabulary blocks, a ragged last one
    torch.manual_seed(0)
    s, b, hdim, v = 37, 1, 16, 50
    h = torch.randn(s, b, hdim, dtype=torch.float64, requires_grad=True)
    w = torch.randn(v, hdim, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, v, (s, b))
    g = torch.randn(s, b, dtype=torch.float64)

    ref = _naive(h, w, y)
    (ref.double() * g).sum().backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None

    out = LinearLogProb.apply(h, w, y, 8, -100)  # chunk 8 -> several chunks incl. a ragged last one
    (out.double() * g).sum().backward()
    assert torch.allclose(out.double(), ref.double(), atol=1e-6)
    assert torch.allclose(h.grad, gh, atol=1e-5)
    assert torch.allclose(w.grad, gw, atol=1e-5)


def test_ignore_index_zero_loss_and_grad():
    h = torch.randn(5, 1, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1], [-100], [3], [-100], [6]])
    out = LinearLogProb.apply(h, w, y, 2, -100)
    assert out[1, 0] == 0 and out[3, 0] == 0
    out.sum().backward()
    assert torch.all(h.grad[1] == 0) and torch.all(h.grad[3] == 0)


def test_install_skips_a_ppo_critic_value_head(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    # install() rebinds Megatron's linear_cross_entropy; a stand-in module (the real one needs a GPU)
    for name in ("megatron", "megatron.core", "megatron.core.transformer", "megatron.core.transformer.linear_cross_entropy"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))

    class Gpt(torch.nn.Module):
        def __init__(self, out):
            super().__init__()
            self.output_layer = torch.nn.Linear(4, out)
            self.fuse_linear_cross_entropy = False
            self.config = SimpleNamespace(cross_entropy_loss_fusion=False, cross_entropy_fusion_impl="native")

    actor, critic = Gpt(8), Gpt(1)
    assert fl.install([actor]) == 1 and actor.fuse_linear_cross_entropy
    assert fl.install([critic]) == 0 and not critic.fuse_linear_cross_entropy


def test_training_forward_leaves_the_exact_entropy(monkeypatch):
    monkeypatch.setattr(fl, "ENTROPY_BLOCK", 16)  # the entropy in 4 vocabulary blocks, a ragged last one
    monkeypatch.setenv("PRL_OBSERVE_ENTROPY", "1")
    torch.manual_seed(1)
    s, b, hdim, v = 37, 1, 16, 50
    h = torch.randn(s, b, hdim, dtype=torch.float64, requires_grad=True)
    w = torch.randn(v, hdim, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, v, (s, b))
    logits = (h.reshape(-1, hdim) @ w.t()).float()
    ref = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1).view(s, b)

    out = fl.linear_cross_entropy(h, w, y)
    ent = fl.take_entropy()
    assert ent is not None and ent.shape == out.shape and not ent.requires_grad
    assert torch.allclose(ent.double(), ref.detach().double(), atol=1e-5)
    assert fl.take_entropy() is None  # taken once
    out.sum().backward()  # the extra input does not disturb the gradient
    assert h.grad is not None and w.grad is not None

    with torch.no_grad():  # the log-prob passes (old policy, reference) leave none
        fl.linear_cross_entropy(h, w, y)
    assert fl.take_entropy() is None
    monkeypatch.setenv("PRL_OBSERVE_ENTROPY", "0")
    fl.linear_cross_entropy(h, w, y)
    assert fl.take_entropy() is None


def test_miles_reads_the_fused_entropy_per_response(monkeypatch):
    """patches/miles/0013: get_log_probs_and_entropy splits the kernel's entropy like the log-probs (CP 1, thd)."""
    from pathlib import Path
    from types import SimpleNamespace

    import pytest

    root = Path(__file__).resolve().parents[2] / "build/miles"
    if not (root / "miles/backends/training_utils/loss/hub/logit_processors.py").exists():
        pytest.skip("Generate build/miles with scripts/setup/apply_patches.sh")
    try:
        monkeypatch.syspath_prepend(str(root))
        from miles.backends.training_utils.loss.hub import logit_processors as lp
    except Exception as e:  # noqa: BLE001 - the host venv lacks Miles' dependencies (any of them); run in the Miles image
        pytest.skip(f"Miles not importable: {e}")
    monkeypatch.setattr(lp, "get_parallel_state", lambda: SimpleNamespace(cp=SimpleNamespace(size=1, rank=0)))
    monkeypatch.setenv("PRL_OBSERVE_ENTROPY", "1")
    torch.manual_seed(2)
    total, resp, hdim, v = [9, 7], [4, 3], 8, 20
    tokens = [torch.randint(0, v, (n,)) for n in total]
    flat = torch.cat(tokens)
    h = torch.randn(sum(total), 1, hdim, dtype=torch.float64, requires_grad=True)
    w = torch.randn(v, hdim, dtype=torch.float64, requires_grad=True)
    labels = torch.roll(flat, -1).view(1, -1)  # as patch 0003: [b, s] next-token labels
    out = fl.linear_cross_entropy(h, w, labels.transpose(0, 1)).transpose(0, 1)  # Megatron's [s, b] round trip
    args = SimpleNamespace(qkv_format="thd", true_on_policy_mode=False, rollout_temperature=1.0, allgather_cp=False)
    res = lp.get_log_probs_and_entropy(out, args=args, unconcat_tokens=tokens, total_lengths=total,
                                       response_lengths=resp, with_entropy=True, entropy_requires_grad=False)
    logits = (h.reshape(-1, hdim) @ w.t()).double()
    ref = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1)
    end = 0
    for i, (t, r) in enumerate(zip(total, resp)):
        end += t
        assert torch.allclose(res["entropy"][i].double(), ref[end - r - 1:end - 1].detach(), atol=1e-5)
        assert torch.allclose(res["log_probs"][i].double(), -out[0, end - r - 1:end - 1].double())


def test_miles_logs_token_distributions_and_explained_variance(monkeypatch):
    """patches/miles/0014: std / min / max over loss-masked tokens and 1 - Var(r - v) / Var(r)."""
    from pathlib import Path
    from types import SimpleNamespace

    import pytest

    root = Path(__file__).resolve().parents[2] / "build/miles"
    try:
        monkeypatch.syspath_prepend(str(root))
        from miles.backends.training_utils.metrics import log_utils as lu
    except Exception as e:  # noqa: BLE001 - the host venv lacks Miles' dependencies (any of them)
        pytest.skip(f"Miles not importable: {e}")
    pg = SimpleNamespace(rank=0, size=1, gloo_groups_inner_to_outer=None)
    monkeypatch.setattr(lu, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=pg, cp=SimpleNamespace(size=1)))
    monkeypatch.setattr(lu, "get_local_response_loss_masks", lambda t, r, m, **k: m)
    monkeypatch.setattr(lu.MultiPGUtil, "gather_object", staticmethod(lambda obj, **k: [obj, obj]))  # two equal ranks
    logged = {}
    monkeypatch.setattr(lu.tracking, "log", lambda args, d, step_key=None: logged.update(d))
    monkeypatch.setattr(lu, "compute_rollout_step", lambda args, rid: rid)
    data = {"total_lengths": [5, 4], "response_lengths": [3, 2],
            "loss_masks": [torch.tensor([1, 1, 0]), torch.tensor([1, 1])],
            "advantages": [torch.tensor([1.0, 1.0, 9.0]), torch.tensor([-1.0, -1.0])],
            "returns": [torch.tensor([1.0, 2.0, 9.0]), torch.tensor([3.0, 4.0])],
            "values": [torch.tensor([1.0, 2.0, 0.0]), torch.tensor([3.0, 3.0])]}
    lu._prl_log_token_distributions(7, SimpleNamespace(qkv_format="thd", trainer_model_id=None), data)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    ret, res = torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0.0, 0.0, 0.0, 1.0])
    get = lambda k: next(v for key, v in logged.items() if key.endswith(k))
    assert abs(get("advantages_std") - adv.std(unbiased=False).item()) < 1e-6
    assert get("advantages_max") == 1.0 and get("returns_min") == 1.0  # the masked 9.0 is left out
    assert abs(get("vf_explained_var") - (1 - res.var(unbiased=False) / ret.var(unbiased=False)).item()) < 1e-6
