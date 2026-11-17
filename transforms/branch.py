"""The branch template: PRL_TRANSFORM=branch.

Every method that lets the policy play a trial again from one of its steps answers three questions; each is a plug-in,
named in the run's settings, built in or the user's own:

    select    where to branch: the steps (anchors), best first
    allocate  how many branches from each anchor
    credit    how a branch trains: ``member`` (one more member of the prompt's group) or ``subgroup`` (only against its
              siblings: the other branches of the step and the trial from the step on; GRPO only)

    PRL_TRANSFORM_ARGS='{"select": "first_bad", "allocate": {"name": "fixed", "n": 2}, "credit": "subgroup"}'

A plug-in is a name or {"name": ..., <its params>}. A name that is not built in is the function of the same name in
``transforms/<name>.py`` (as for transforms): ``select(traj, **params)`` returns steps (``Step`` or 0-based index),
optionally as (step, score) pairs; ``allocate(traj, anchors, **params)`` returns (step, n) pairs, n 0 for none.

Built in:
    select    middle      the middle turn (framework tests)
              teacher     the teacher reads the trial, picks the step and may correct the turns before it
                          (transforms/rewrite.py ``teacher_choice``; params prompt, obs_chars, max_change_share)
              first_bad   the judge's first bad step (prl/algo/labeler.py ``grade``)
              surprisal   the steps where the policy's own tokens were least likely: mean −logprob of the step's sampled
                          tokens (``rollout_log_probs``), a stand-in for entropy; param ``top`` (default 1)
    allocate  fixed       ``n`` branches per anchor (default 1); ``only``: "failed" (default; reward 0), "passed"
                          (reward > 0) or "all", checked before select, so a trial left out costs no model call;
                          ``view``: "trial" (decide from this trial alone, default) or "group" (from the whole group:
                          planned, refused for now)

Anchors before ``min_step`` (default 1: from step 0 a branch is just one more trial) are dropped; at most
``max_anchors`` (default 1) are kept, in the order select gave (best first). Trials split by a summary cannot branch
and are left as they are. What was chosen is in ``traj.record["branch"]``; the branches' outcomes in the trial's
transform record (prl/transforms/api.py, prl/rollout/branching.py). The run needs the reroll arm
(``ota_train_transform_reroll``: the group-reward post-process).
"""

from __future__ import annotations

import inspect
from typing import Any

from prl.algo import labeler
from prl.transforms import Step, Trajectory
from prl.transforms.core import _resolve

VIEWS = ("trial", "group")
ONLY = ("failed", "passed", "all")


async def select_middle(traj: Trajectory) -> list[Step]:
    return [traj.steps[len(traj.steps) // 2]]


async def select_teacher(traj: Trajectory, prompt: str = "reroll_v1", obs_chars: int = 3000,
                         max_change_share: float | None = None) -> list[int]:
    from transforms.rewrite import teacher_choice  # a sibling transform: imported when used

    k = await teacher_choice(traj, prompt, obs_chars, max_change_share)
    return [] if k is None else [k]


async def select_first_bad(traj: Trajectory) -> list[int]:
    grades = await labeler.grade(traj)
    if grades is None:
        traj.skip("the judge gave no usable answer (atlas_labels.json)")
        return []
    return sorted(grades[1])[:1]


async def select_surprisal(traj: Trajectory, top: int = 1) -> list[tuple[int, float]]:
    scores = []
    for step in traj.steps:
        lp = getattr(traj.samples[step._sample], "rollout_log_probs", None)
        if not lp:
            raise ValueError("surprisal needs the samples' rollout_log_probs")
        part = lp[step._start:step._end]
        scores.append((step.index, -sum(part) / max(len(part), 1)))
    return sorted(scores, key=lambda x: -x[1])[:top]


async def allocate_fixed(traj: Trajectory, anchors: list[Step], n: int = 1, only: str = "failed",
                         view: str = "trial") -> list[tuple[Step, int]]:
    return [(a, n) for a in anchors]


SELECT = {"middle": select_middle, "teacher": select_teacher, "first_bad": select_first_bad,
          "surprisal": select_surprisal}
ALLOCATE = {"fixed": allocate_fixed}


def _plugin(spec: Any, builtin: dict[str, Any], kind: str) -> tuple[str, Any, dict[str, Any]]:
    """(name, function, params) of a plug-in given as a name or {"name": ..., params}."""
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict) and isinstance(spec.get("name"), str):
        name, params = spec["name"], {k: v for k, v in spec.items() if k != "name"}
    else:
        raise ValueError(f"{kind} must be a name or {{'name': ..., params}}, got {spec!r}")  # noqa: TRY004 -- a config value
    fn = builtin.get(name) or _resolve(name)
    return name, fn, params


async def _call(fn: Any, *args: Any, **params: Any) -> Any:
    out = fn(*args, **params)
    return await out if inspect.isawaitable(out) else out


def _step(traj: Trajectory, x: Any) -> Step:
    step = x if isinstance(x, Step) else traj.steps[int(x)]
    if not any(step is s for s in traj.steps):
        raise ValueError(f"select returned a step of another trial: {x!r}")
    return step


async def branch(traj: Trajectory, select: Any = "middle", allocate: Any = "fixed", credit: str = "member",
                 min_step: int = 1, max_anchors: int = 1) -> Trajectory:
    s_name, s_fn, s_params = _plugin(select, SELECT, "select")
    a_name, a_fn, a_params = _plugin(allocate, ALLOCATE, "allocate")
    if a_fn is allocate_fixed:
        view, only = a_params.get("view", "trial"), a_params.get("only", "failed")
        if view not in VIEWS or only not in ONLY:
            raise ValueError(f"allocate fixed: view must be one of {VIEWS} and only one of {ONLY}, got {view!r}, "
                             f"{only!r}")
        if view == "group":
            raise ValueError("allocate view 'group' (decide from the whole group) is planned, not implemented yet; "
                             "use 'trial'")
        failed = traj.reward is not None and traj.reward <= 0
        if traj.reward is None or (only == "failed" and not failed) or (only == "passed" and failed):
            traj.record["branch"] = {"why": f"reward {traj.reward} left out (only {only})"}
            return traj
    record = traj.record["branch"] = {"select": s_name, "allocate": a_name, "credit": credit}
    if len(traj.samples) > 1:
        record["why"] = "split by a summary"
        return traj
    picked = await _call(s_fn, traj, **s_params)
    if traj.skipped is not None:
        return traj
    anchors: list[Step] = []
    for x in picked or []:
        step = _step(traj, x[0] if isinstance(x, tuple) else x)
        if step.index >= min_step and not any(step is a for a in anchors):
            anchors.append(step)
    anchors = anchors[:max_anchors]
    plan = await _call(a_fn, traj, anchors, **a_params)
    record["anchors"] = []
    for step, n in plan:
        step = _step(traj, step)
        if n:
            step.reroll(int(n), credit)
            record["anchors"].append({"step": step.index, "n": int(n)})
    if picked and not anchors:
        record["why"] = f"every selected step was before min_step {min_step}"
    return traj
