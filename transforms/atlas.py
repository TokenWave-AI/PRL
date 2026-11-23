"""ATLAS (ATLaS: Agent Tuning via Learning Critical Steps, Chen et al., ACL 2025, arXiv 2503.02197) as a trajectory transform: PRL_TRANSFORM=atlas.

A judge model marks the trial's critical steps (prl/algo/labeler.py, configured by PRL_JUDGE_*). The other
steps leave the loss (``other_weight`` 0, the default: hard mask) or have their advantage scaled by ``other_weight``
(ATLAS-soft, e.g. PRL_TRANSFORM_ARGS='{"other_weight": 0.3}'). A trial the judge could not label (HTTP
errors, unparsable answers after the retries) is skipped: it trains with all its steps, as plain GRPO. The judge's
request and answer are in ``<trial_dir>/atlas_labels.json``; the critical steps also in ``traj.record["atlas"]``
(numbered from 1, as in the judge's prompt and atlas_labels.json; ``step.index`` counts from 0).

``select="random"`` is the control for the judge's choice: no judge call, and random steps stand in for the critical
ones. They hold about ``keep_tokens`` of the trial's trained tokens: the default 0.317 is the share a signed ATLAS run of
ours kept (its critical steps were 22.8% of the steps), so the control trains as many tokens, not as many steps
(critical steps are longer than average). The steps are drawn in an order fixed by
``seed`` and the trial directory's name, each taken while it brings the share closer to ``keep_tokens``; the first that
would not ends the draw (no preference for short steps), and at least one step is kept. The metrics are the judge's
(rollout/atlas/critical_step_frac and kept_token_frac); the record also says ``"select": "random"``.
"""

from __future__ import annotations

import random
from pathlib import Path

from prl.algo import labeler
from prl.transforms import Trajectory

SELECT = ("judge", "random")


async def atlas(traj: Trajectory, other_weight: float = 0.0, select: str = "judge", keep_tokens: float = 0.317,
                seed: int = 0) -> Trajectory:
    if select not in SELECT:
        raise ValueError(f"select must be one of {SELECT}, got {select!r}")
    if not 0 < keep_tokens <= 1:
        raise ValueError(f"keep_tokens is a share of the trial's trained tokens, in (0, 1], got {keep_tokens}")
    critical = await labeler.label(traj) if select == "judge" else _random_steps(traj, keep_tokens, seed)
    traj.record["atlas"] = {"steps": len(traj.steps),
                            "critical": None if critical is None else sorted(n + 1 for n in critical)}
    if select == "random":
        traj.record["atlas"]["select"] = "random"
    if critical is None:
        traj.skip("the judge gave no usable answer (atlas_labels.json)")
        return traj
    for step in traj.steps:
        if step.index in critical:
            continue
        if other_weight == 0.0:
            step.mask()
        else:
            step.weight = other_weight
    return traj


def _random_steps(traj: Trajectory, keep_tokens: float, seed: int) -> set[int]:
    """0-based indices of random steps holding about ``keep_tokens`` of the trial's trained tokens (at least one)."""
    # A step's trained tokens, as transforms/branch.py select_surprisal reads them
    size = {step.index: step._end - step._start for step in traj.steps}
    order = sorted(size)
    # A str seed is hashed the same way in every process (unlike hash()); the trial directory tells trials apart
    # the trial directory's name, not its path: the same trial draws the same steps wherever its directory sits
    random.Random(f"{seed}:{Path(traj.trial_dir).name if traj.trial_dir else traj.task}:{len(order)}").shuffle(order)
    target, kept, total = keep_tokens * sum(size.values()), set(), 0
    for n in order:
        if kept and abs(total + size[n] - target) >= abs(total - target):
            break
        kept.add(n)
        total += size[n]
    return kept
