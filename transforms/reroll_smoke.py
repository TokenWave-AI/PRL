"""Framework smoke of ``step.reroll()`` (developers only; prl/transforms/api.py, prl/rollout/branching.py): no
teacher, so the smoke tests the replay and the added group member, not a judgment.

Every trial in one piece with at least ``min_steps`` turns rerolls from its middle turn, ``n`` times, trained as
``credit`` says; ``second`` also from its last turn (two anchors). With ``edit``, the turn before the middle one gets a
note appended to its reasoning: a rewrite before the rerolled turn must reach the replay only (the trial trains as
generated, scripts/checks/grpo_acceptance.py). ``every`` > 1 rerolls one trial in that many, picked by a hash of the
trial directory, to keep a smoke short.
"""

from __future__ import annotations

import hashlib

from prl.transforms import Trajectory

NOTE = "\n(The run's reroll smoke appended this sentence.)"


def reroll_smoke(traj: Trajectory, min_steps: int = 3, edit: bool = True, every: int = 1, n: int = 1,
                 credit: str = "member", second: bool = False) -> Trajectory:
    if len(traj.samples) > 1 or len(traj.steps) < min_steps:
        return traj  # left as it is on purpose: core refuses a reroll of a summarized trial
    if every > 1 and int(hashlib.sha256(str(traj.trial_dir).encode()).hexdigest()[:8], 16) % every:
        return traj
    at = traj.steps[len(traj.steps) // 2]
    before = traj.steps[at.index - 1]
    if edit and before.thinking is not None:
        before.set_thinking(before.thinking + NOTE)
    at.reroll(n, credit)
    if second and traj.steps[-1].index > at.index:
        traj.steps[-1].reroll(n, credit)
    traj.record["reroll_smoke"] = {"steps": len(traj.steps), "at": at.index, "edited": edit, "n": n, "credit": credit,
                                   "second": second}
    return traj
