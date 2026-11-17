"""ATLAS (Chen et al., ACL Findings 2025, arXiv 2503.02197) as a trajectory transform: PRL_TRANSFORM=atlas.

A judge model marks the trial's critical steps (prl/algo/labeler.py, configured by PRL_JUDGE_*). The other
steps leave the loss (``other_weight`` 0, the default: hard mask) or have their advantage scaled by ``other_weight``
(ATLAS-soft, e.g. PRL_TRANSFORM_ARGS='{"other_weight": 0.3}'). A trial the judge could not label (HTTP
errors, unparsable answers after the retries) is skipped: it trains with all its steps, as plain GRPO. The judge's
request and answer are in ``<trial_dir>/atlas_labels.json``; the critical steps also in ``traj.record["atlas"]``.
"""

from __future__ import annotations

from prl.algo import labeler
from prl.transforms import Trajectory


async def atlas(traj: Trajectory, other_weight: float = 0.0) -> Trajectory:
    critical = await labeler.label(traj)
    traj.record["atlas"] = {"steps": len(traj.steps),
                            "critical": None if critical is None else sorted(n + 1 for n in critical)}
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
