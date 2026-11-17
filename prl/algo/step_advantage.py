"""Step-level advantage weighting for agentic RL (Miles hook).

    --custom-advantage-postprocess-path prl.algo.step_advantage.step_weight_postprocess

Runs right after Miles' built-in estimator (e.g. GRPO, whose advantage is the
group-normalized outcome reward broadcast to every response token) and rescales
it per agent *step*:

    A[t] = w[step(t)] * A_estimator[t]

Steps are recovered from the response ``loss_mask``: each maximal run of 1s is
one assistant turn. With Miles' TITO session server, every model call's sampled
tokens are kept verbatim (loss_mask 1) and every tool observation or template
glue is 0, so run k is model call k, which is mimoagent step k. This is the
same indexing as ``tool_call_error_flags``.

Weights come from ``sample.train_metadata["step_weights"]`` (a list, one float
per step), which Miles ships to the trainer as ``rollout_data["metadata"]``
(patches/miles/0001). With no weights the function is the identity, i.e. plain
GRPO, which is our baseline arm. Trajectory transforms (``step.weight``,
prl/transforms/core.py, e.g. ATLAS-soft) and error/repair signals set them.

A sample with ``train_metadata["group_reward"]`` (``Trajectory.add_sample``) keeps its advantage, but under PPO its value
target is the critic's own value: that reward is a model's judgment, not an outcome, and the critic learns outcomes only.

An optional ``train_metadata["turn_ids"]`` (per response token, 1-based turn
index, -1 elsewhere) overrides the loss_mask segmentation.

Token level: ``train_metadata["token_weight_spans"]`` (a list of [start, end, w], response token ranges; written by
``step.weight_text``, prl/transforms/core.py) gives those tokens the weight ``w`` instead of their step's, e.g.
only the wrong command of a bad step is pushed down. A sample with spans and no step weights has weight 1 elsewhere.

Subgroup credit: ``train_metadata["advantage_spans"]`` (a list of [start, end, value], response token ranges; written by
prl/algo/added_rewards.py for a trial whose step was rerolled with ``credit="subgroup"``) sets those tokens' advantage
(and, outside PPO, return) to ``value`` before the weights act: from that step on the trial is compared with its branches,
not with its group.

Under PPO (``--advantage-estimator ppo``) only the advantages are scaled: the
returns are the critic's targets and stay as GAE computed them. The weights
act before Miles' optional ``--normalize-advantages`` whitening, so with it on a
weight scales the advantage before the batch is re-centred (approximate).

Under context parallelism the advantages are this rank's zigzag chunks of the response; the weights are
computed on the whole response and sliced the same way as Miles slices the loss masks.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def loss_mask_steps(loss_mask: torch.Tensor) -> torch.Tensor:
    """Map each response token to its step index (0-based); -1 where loss_mask == 0."""
    m = loss_mask.to(torch.int64)
    starts = (m == 1) & torch.cat([m.new_ones(1), (m[:-1] == 0).to(torch.int64)]).bool()
    step = torch.cumsum(starts.to(torch.int64), dim=0) - 1
    return torch.where(m == 1, step, torch.full_like(step, -1))


def token_step_weights(md: dict, loss_mask: torch.Tensor, n: int, device) -> torch.Tensor | None:
    """Per-token weights (length n, aligned to the tail of the response) or None."""
    weights = md.get("step_weights")
    spans = md.get("token_weight_spans")
    if not weights:
        return _apply_spans(torch.ones(n, device=device), spans) if spans else None
    return _apply_spans(_step_weights(md, weights, loss_mask, n, device), spans)


def _apply_spans(tw: torch.Tensor, spans) -> torch.Tensor:
    """Token weight spans (response token ranges, the response being the last n tokens) override the step weights."""
    for start, end, w in spans or []:
        tw[max(0, int(start)):min(tw.numel(), int(end))] = float(w)
    return tw


def _step_weights(md: dict, weights, loss_mask: torch.Tensor, n: int, device) -> torch.Tensor:
    if md.get("turn_ids") is not None:
        tid = torch.as_tensor(md["turn_ids"][-n:], dtype=torch.int64, device=device)
        steps = torch.where(tid > 0, tid - 1, torch.full_like(tid, -1))
    else:
        steps = loss_mask_steps(torch.as_tensor(loss_mask)[-n:].to(device))
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    idx = steps.clamp(min=0, max=max(w.numel() - 1, 0))
    # Steps beyond the provided weights (e.g. labeler truncation) and non-step tokens keep weight 1.
    return torch.where((steps >= 0) & (steps < w.numel()), w[idx], torch.ones(n, device=device))


def _set_spans(x: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
    """``x`` where ``full`` (this rank's part of the spans' values, NaN outside them) has none."""
    return torch.where(torch.isnan(full), x, full.to(x.dtype))


def apply_step_weights(advantages, returns, loss_masks, metadata, to_local=lambda i, x: x, scale_returns=True,
                       values=None):
    """to_local(i, x): sample i's per-response-token tensor x -> this rank's part (identity without CP).
    scale_returns: False when the returns are value targets (PPO). values: the critic's, per sample (PPO)."""
    out_adv, out_ret = [], []
    n_weighted = n_tok = n_zeroed = n_spanned = 0
    for i, adv in enumerate(advantages):
        md = metadata[i] if metadata is not None and isinstance(metadata[i], dict) else {}
        if values is not None and md.get("group_reward") is not None:
            out_adv.append(adv)
            out_ret.append(values[i].detach().clone())
            continue
        lm = torch.as_tensor(loss_masks[i]).to(adv.device)
        ret = returns[i]
        if md.get("advantage_spans") and values is None:
            full = torch.full((lm.numel(),), float("nan"), device=adv.device)
            for start, end, value in md["advantage_spans"]:
                full[max(0, int(start)):min(full.numel(), int(end))] = float(value)
            full = to_local(i, full)
            adv = _set_spans(adv, full)
            ret = _set_spans(ret, full) if scale_returns else ret
            n_spanned += 1
        tw = token_step_weights(md, lm, lm.numel(), adv.device)
        if tw is None:
            out_adv.append(adv)
            out_ret.append(ret)
            continue
        tw, lm = to_local(i, tw), to_local(i, lm).bool()
        out_adv.append(adv * tw)
        out_ret.append(ret * tw if scale_returns else ret)
        n_weighted += 1
        n_tok += int(lm.sum())
        n_zeroed += int(((tw == 0) & lm).sum())
    logger.info("[prl] step weights: %d/%d samples weighted; %d/%d trained tokens zero-weighted; %d with subgroup "
                "advantage spans", n_weighted, len(advantages), n_zeroed, n_tok, n_spanned)
    return out_adv, out_ret


def step_weight_postprocess(args, rollout_data, advantages, returns):
    """Miles hook (prl patch): (args, rollout_data, advantages, returns) -> (advantages, returns)."""
    from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks

    total, response, max_seq = rollout_data["total_lengths"], rollout_data["response_lengths"], rollout_data.get("max_seq_lens")

    def to_local(i, x):
        return get_local_response_loss_masks([total[i]], [response[i]], [x], args.qkv_format,
                                             None if max_seq is None else [max_seq[i]])[0]

    return apply_step_weights(advantages, returns, rollout_data["loss_masks"], rollout_data.get("metadata"), to_local,
                              scale_returns=getattr(args, "advantage_estimator", "grpo") != "ppo",
                              values=rollout_data.get("values"))
