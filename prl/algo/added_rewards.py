"""The rewards the estimator trains on (Miles: --custom-reward-post-process-path). The raw rewards stay the verifier's, so the reward
metric is unchanged.

Length: with PRL_LENGTH_LAMBDA > 0, a passing trial's reward is 1 - lambda * min(1, tokens / PRL_LENGTH_CAP), tokens being
its model output (loss_mask); a failed trial stays 0, so giving up early earns nothing.

GRPO: as Miles' train_data_conversion._normalize_rewards_by_rollout, within a prompt's group, a rollout's reward minus the group's mean,
over its std; PRL_REWARD_STD_FLOOR is a lower bound on that std, else a tiny difference of lengths in an all-pass group would
weigh as much as passing against failing. Without length and floor the result is Miles'.
A sample with ``train_metadata["group_reward"]`` (``Trajectory.add_sample``, a reroll: prl/rollout/harbor_training.py) is one more
member of the group, with that reward; samples of one rollout with the same ``train_metadata["added_id"]`` are one member together.

Subgroup credit (``step.reroll(credit="subgroup")``, prl/transforms/core.py): a branch (``train_metadata["branch"]``,
``credit`` subgroup) is not a member of the group. It is normalized with its siblings only: the other branches of the same step
of the same trial, and that trial, whose ``train_metadata["branch_anchors"]`` lists the step and its first response token. The
trial keeps its group value; its tokens from the step on get the siblings' value instead, written as
``train_metadata["advantage_spans"]`` ([start, end, value], response tokens; prl/algo/step_advantage.py sets them), from each
anchor to the next. Siblings that all scored the same give 0 from the step on: the step is compared with its alternatives only, so
the trial's own outcome no longer pushes those tokens. GRPO only (PPO's advantage is the critic's).

Verified credit (``credit="verified"``; transforms/hint_reroll.py): the branches and the trial from the
step on are siblings as with subgroup, but what the trial's tokens get depends on the outcome. When a branch scored more
than the trial, the state at the step could still succeed: the trial's tokens before the step get 0 (not blamed), from
the step on the siblings' value. When none did, nothing confirms the step as the mistake: the trial keeps its group value
throughout, and the branches take their siblings' value (0 when they tied with the trial). GRPO only.

PPO has no group: an added sample's reward goes in as it is, against the critic's values; the others as the length rule gives them.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

import torch

from prl.transforms.api import SIBLING_CREDITS

logger = logging.getLogger(__name__)


def _shaped(raw: list[float], samples: list[Any]) -> list[float]:
    lam = float(os.getenv("PRL_LENGTH_LAMBDA", "0"))
    tokens: dict[Any, int] = defaultdict(int)
    added = [(s.train_metadata or {}).get("group_reward") is not None for s in samples]
    for s, is_added in zip(samples, added, strict=True):
        if not is_added:
            tokens[s.rollout_id] += sum(s.loss_mask)
    passed = {s.rollout_id: float(r) for s, r, a in zip(samples, raw, added, strict=True) if not a}
    wins = sum(1 for r in passed.values() if r >= 1.0)
    logger.info("[prl] model output tokens: %.0f per trial, %.0f per pass, %.0f per verifier point (%d trials, %d pass)",
                sum(tokens.values()) / max(len(tokens), 1),
                sum(tokens[k] for k, r in passed.items() if r >= 1.0) / max(wins, 1),
                sum(tokens.values()) / max(sum(passed.values()), 1e-9), len(tokens), wins)
    if lam <= 0:
        return [float(r) for r in raw]
    cap = float(os.getenv("PRL_LENGTH_CAP", "8192"))
    return [float(r) * (1 - lam * min(1.0, tokens[s.rollout_id] / cap)) for s, r in zip(samples, raw, strict=True)]


def _normalized(rewards: list[float], args: Any, floor: float) -> list[float]:
    """As Miles' GRPO: minus the mean, over the std (with the floor) when --grpo-std-normalization."""
    r = torch.tensor(rewards, dtype=torch.float)
    out = r - r.mean()
    if args.grpo_std_normalization and len(rewards) > 1 and (floor > 0 or r.std() > 0):
        out = out / (max(float(r.std()), floor) + 1e-6)
    return out.tolist()


def group_rewards(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    raw = [s.get_reward_value(args) for s in samples]
    shaped = _shaped(raw, samples)
    subgroup = [((s.train_metadata or {}).get("branch") or {}).get("credit") in SIBLING_CREDITS for s in samples]
    if args.advantage_estimator == "ppo":
        if any(subgroup):
            raise ValueError("subgroup credit (step.reroll) is for GRPO: under PPO the critic gives the advantage")
        added = [(s.train_metadata or {}).get("group_reward") for s in samples]
        return raw, [r if a is None else float(a) for r, a in zip(shaped, added, strict=True)]
    floor = float(os.getenv("PRL_REWARD_STD_FLOOR", "0"))
    groups: dict[int, dict[Any, tuple[float, list[int]]]] = {}
    # (group, rollout, step) -> member -> (reward, rows); the trial itself is member "trial"
    siblings: dict[tuple[int, Any, int], dict[Any, tuple[float, list[int]]]] = {}
    starts: dict[tuple[int, Any, int], int] = {}
    credits: dict[tuple[int, Any, int], str] = {}
    for i, s in enumerate(samples):
        if s.group_index is None or s.rollout_id is None:
            raise ValueError(f"sample {i} needs a group_index and a rollout_id")
        tm = s.train_metadata or {}
        added = tm.get("group_reward")
        # An added trial split into several samples (a reroll with a summary: train_metadata "added_id") is one member
        member_id = tm.get("added_id", i)
        key, reward = (("added", s.rollout_id, member_id), float(added)) if added is not None else (s.rollout_id, shaped[i])
        if subgroup[i]:
            table = siblings.setdefault((int(s.group_index), s.rollout_id, int(tm["branch"]["anchor"])), {})
        else:
            table = groups.setdefault(int(s.group_index), {})
            for anchor in tm.get("branch_anchors") or []:
                at = (int(s.group_index), s.rollout_id, int(anchor["step"]))
                siblings.setdefault(at, {}).setdefault("trial", (reward, []))[1].append(i)
                starts[at] = int(anchor["start"])
                credits[at] = anchor.get("credit", "subgroup")
        member = table.setdefault(key, (reward, []))
        if member[0] != reward:
            raise ValueError(f"all samples of rollout {key!r} must share one reward: {member[0]}, {reward}")
        member[1].append(i)
    out = [0.0] * len(samples)
    for members in groups.values():
        for (_, rows), value in zip(members.values(), _normalized([m[0] for m in members.values()], args, floor),
                                    strict=True):
            for i in rows:
                out[i] = value
    spans: dict[int, list[tuple[int, float]]] = {}
    for at, members in siblings.items():
        if "trial" not in members or len(members) < 2:
            # A branch whose trial is not in the batch, or a step without a branch left: nothing to compare with
            raise ValueError(f"subgroup at {at}: needs the trial and at least one branch, got {sorted(map(str, members))}")
        verified = credits[at] == "verified"
        confirmed = any(name != "trial" and m[0] > members["trial"][0] for name, m in members.items())
        for (name, (_, rows)), value in zip(members.items(), _normalized([m[0] for m in members.values()], args, floor),
                                            strict=True):
            for i in rows:
                if name != "trial":
                    out[i] = value
                elif not verified:
                    spans.setdefault(i, []).append((starts[at], value))
                elif confirmed:  # the steps before this one are not blamed: 0 from the response's start
                    marks = spans.setdefault(i, [])
                    marks.append((starts[at], value))
                    if not any(a == 0 for a, _ in marks):
                        marks.append((0, 0.0))
                # verified, not confirmed: the trial keeps its group value
    for i, marks in spans.items():
        marks.sort()
        end = [m[0] for m in marks[1:]] + [samples[i].response_length]
        samples[i].train_metadata = {**samples[i].train_metadata,
                                     "advantage_spans": [[a, b, v] for (a, v), b in zip(marks, end, strict=True)]}
    return raw, out
