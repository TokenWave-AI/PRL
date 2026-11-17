"""Memory-lean output layer + token log-prob for long agentic trajectories (Hopper and older).

Problem: Miles computes the policy log-prob from full-vocabulary logits. For Qwen3.5 (vocab
248,320) a 64k-token trajectory needs the bf16 logits (32 GB), Megatron's fused cross-entropy keeps
an fp32 softmax for backward (64 GB), plus the logits gradient: far more than a 96 GB GH200 has next
to the model. Chunking the log-prob (``--log-probs-chunk-size``) only lowers the transient peak;
autograd still keeps every chunk's activations.

Fix: never materialize the full logits. ``LinearLogProb`` takes the final hidden states and the
output weight, and per chunk of ``PRL_FUSED_LOGPROB_CHUNK`` tokens (default 4096) computes
``logits = h @ W^T`` (bf16 matmul, fp32 math), ``lse`` and ``-logprob[label]``. Only ``h`` and the
per-token ``lse`` are saved; backward recomputes each chunk's logits:
``dlogits = softmax - onehot(label)`` scaled by the incoming gradient, then
``dh = dlogits @ W`` and ``dW += dlogits^T @ h``. Peak extra memory is ~2 chunks of logits,
independent of sequence length.

It is installed as the implementation of Megatron's ``linear_cross_entropy`` (whose shipped kernel
is Blackwell-only), so ``GPTModel.forward(labels=...)`` returns per-token ``-log p(label)``; the Miles
patch (patches/miles/0003) passes next-token labels and consumes these log-probs directly.
Supports TP=1 without sequence parallelism (our single-GPU setup), and rollout temperature 1.0.

Entropy (``PRL_OBSERVE_ENTROPY=1``, set with Miles' ``--observe-training-entropy``): in a forward with grad
(the training step; not the no-grad log-prob passes), each chunk also gives the exact entropy of the full softmax,
``H = lse - sum_v p_v z_v``, in blocks of the vocabulary (one [chunk, 8192] fp32 temporary), detached. It
is left in ``take_entropy()`` for the Miles patch (patches/miles/0013), which reads it in place of the zeros the fused
path used to report as ``train/entropy_loss``. For logging only: no gradient flows through it.
"""

from __future__ import annotations

import os

import torch

# Vocabulary rows per block of the output weight's gradient (32768 x 4096 bf16 = 256 MB)
VOCAB_BLOCK = 32768
# Vocabulary columns per block of the entropy
ENTROPY_BLOCK = 8192
# Entropy of the last forward with grad, shaped like its log-prob output; taken (and cleared) by take_entropy()
_last_entropy = None


def observe_entropy() -> bool:
    return os.environ.get("PRL_OBSERVE_ENTROPY", "0") == "1"


def take_entropy():
    global _last_entropy
    ent, _last_entropy = _last_entropy, None
    return ent


def _chunk_entropy(logits: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits) per row, ``lse - sum_v p_v z_v``, in blocks of ENTROPY_BLOCK vocabulary columns:
    one [rows, ENTROPY_BLOCK] fp32 temporary (2048 x 8192 = 64 MB)."""
    pz = torch.zeros_like(lse)
    for v0, v1 in _chunks(logits.size(1), ENTROPY_BLOCK):
        z = logits[:, v0:v1]
        p = torch.sub(z, lse.unsqueeze(1)).exp_()
        pz += p.mul_(z).sum(-1)
        del p
    return lse - pz


def _chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(n, s + size)


class LinearLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk: int, ignore_index: int,
                with_entropy: bool = False):
        shape = labels.shape
        x = hidden.reshape(-1, hidden.size(-1))
        y = labels.reshape(-1)
        assert x.size(0) == y.size(0), f"{tuple(hidden.shape)} vs {tuple(labels.shape)}"
        n = x.size(0)
        loss = torch.empty(n, dtype=torch.float32, device=x.device)
        lse = torch.empty(n, dtype=torch.float32, device=x.device)
        ent = torch.empty(n, dtype=torch.float32, device=x.device) if with_entropy else None
        for s, e in _chunks(n, chunk):
            logits = (x[s:e] @ weight.t()).float()
            lse_c = torch.logsumexp(logits, dim=-1)
            yc = y[s:e].clamp(min=0)
            tgt = logits.gather(1, yc.unsqueeze(1)).squeeze(1)
            loss[s:e] = lse_c - tgt
            lse[s:e] = lse_c
            if ent is not None:
                ent[s:e] = _chunk_entropy(logits, lse_c)
            del logits
        if ignore_index is not None:
            loss = loss.masked_fill(y == ignore_index, 0.0)
        if ent is not None:
            global _last_entropy
            _last_entropy = ent.view(shape)
        ctx.save_for_backward(hidden, weight, labels, lse)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        return loss.view(shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        hidden, weight, labels, lse = ctx.saved_tensors
        x = hidden.reshape(-1, hidden.size(-1))
        y = labels.reshape(-1)
        g = grad_out.reshape(-1).float()
        if ctx.ignore_index is not None:
            g = g.masked_fill(y == ctx.ignore_index, 0.0)
        grad_x = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if ctx.needs_input_grad[1] else None
        for s, e in _chunks(x.size(0), ctx.chunk):
            logits = (x[s:e] @ weight.t()).float()
            p = torch.exp(logits.sub_(lse[s:e].unsqueeze(1)))  # softmax, in place of logits
            idx = torch.arange(e - s, device=x.device)
            p[idx, y[s:e].clamp(min=0)] -= 1.0  # d(lse - logit_y)/dlogits
            p.mul_(g[s:e].unsqueeze(1))
            p = p.to(x.dtype)
            if grad_x is not None:
                grad_x[s:e] = p @ weight
            if grad_w is not None:
                # in-place add casts the bf16 product while accumulating: no vocab x hidden fp32 temporary (3.8 GB
                # for Qwen3.5's 248k vocab; it pushed a run into a CUDA OOM at step 2). In blocks of vocabulary
                # rows: the bf16 product of the whole vocabulary is 1.9 GB more (a PPO actor next to its critic OOMed
                # on it); rows are independent, so the values are the same.
                pt = p.t()
                for v0, v1 in _chunks(pt.size(0), VOCAB_BLOCK):
                    grad_w[v0:v1].add_(pt[v0:v1] @ x[s:e])
            del p, logits
        return (grad_x.view_as(hidden) if grad_x is not None else None,
                grad_w.to(weight.dtype) if grad_w is not None else None, None, None, None, None)


def linear_cross_entropy(hidden, weight, labels, sequence_parallel=False, reduction="none", ignore_index=-100,
                         tp_group=None, **_):
    """Drop-in for ``megatron.core.fusions.fused_linear_cross_entropy.linear_cross_entropy`` (TP=1)."""
    assert not sequence_parallel, "prl fused log-prob: sequence parallelism not supported"
    assert tp_group is None or tp_group.size() == 1, "prl fused log-prob: TP must be 1"
    chunk = int(os.environ.get("PRL_FUSED_LOGPROB_CHUNK", "4096"))
    # A forward with grad is the training step (the log-prob passes run under no_grad): only it logs the entropy
    loss = LinearLogProb.apply(hidden, weight, labels, chunk, ignore_index, observe_entropy() and torch.is_grad_enabled())
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def install(model_chunks) -> int:
    """Route every GPTModel in ``model_chunks`` through ``linear_cross_entropy`` when labels are given; returns how
    many (0 for a PPO critic, whose value head is left alone)."""
    import megatron.core.transformer.linear_cross_entropy as lce

    lce.linear_cross_entropy = linear_cross_entropy
    if os.environ.get("PRL_FREEZE_EMBEDDINGS") == "1":
        # Experiment: freeze the (tied) input embedding / output projection. With a 248k vocabulary every row gets
        # a tiny but consistently signed softmax gradient, which Adam's per-parameter normalisation turns into a
        # full-size lr step for all rows (mass leaks to the vocabulary tail). With requires_grad off, the rows get
        # no gradient, so Adam's moments stay 0 and the update is 0.
        n_frozen = 0
        for chunk in model_chunks:
            for name, prm in chunk.named_parameters():
                if "word_embeddings.weight" in name or name.endswith("output_layer.weight"):
                    prm.requires_grad_(False)
                    n_frozen += 1
        print(f"[prl] froze {n_frozen} embedding/output parameter tensor(s)", flush=True)
    n = 0
    for chunk in model_chunks:
        for m in chunk.modules():
            if hasattr(m, "fuse_linear_cross_entropy") and hasattr(m, "output_layer"):
                head = m.output_layer
                if isinstance(head, torch.nn.Linear) and head.out_features == 1:
                    continue  # a PPO critic's value head (Miles LinearForLastLayer): values, not log-probs
                m.fuse_linear_cross_entropy = True
                m.config.cross_entropy_loss_fusion = True
                m.config.cross_entropy_fusion_impl = "linear"
                n += 1
    return n


def enabled() -> bool:
    return os.environ.get("PRL_FUSED_LOGPROB", "0") == "1"
