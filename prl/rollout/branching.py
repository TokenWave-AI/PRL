"""Branches: the trials a transform asked for with ``step.reroll(n, credit)`` (prl/transforms/api.py), run by
prl/rollout/harbor_training.py ``generate`` right after the transform chain on a finished training trial.

For each rerolled step, ``n`` more trials (branches) of the same prompt whose first turns replay the trial's
(prl/rollout/reroll_agent.py), all at once, each trained with its own verifier reward (train_metadata
``group_reward`` and ``branch``, prl/algo/added_rewards.py, which the run must use as
--custom-reward-post-process-path). Their outcomes (a list, one per branch) replace the request in the trial's
``prl_transform`` record; each branch has ``reroll/s<step>_b<j>/`` in the trial directory (request, outcome,
replay comparison). A failed branch leaves the trial without it; an aborted one runs again first (as an aborted trial,
PRL_ABORT_MAX_RETRIES), and with PRL_REJECT_MISMATCH one that fails the TITO check is dropped, not retried.

Configuration:
    PRL_REROLL_TIMEOUT: seconds a reroll may take (default 0: only the trial's own Harbor limits); after it the
        trial trains without its reroll. Asynchronous training: a group waits for its rerolls, so set it below the
        update interval; the group's staleness (--max-weight-staleness) is its oldest trial's, the reroll's calls are
        newer.
    PRL_REROLL_CONCURRENCY: rerolls running at once per rollout worker (default 32), besides the trials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput

logger = logging.getLogger(__name__)
ADDED_REWARDS = "prl.algo.added_rewards.group_rewards"


_reroll_slots: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


async def _reroll(input: GenerateFnInput, result: GenerateFnOutput, pristine: Any, rollout_id: int) -> GenerateFnOutput:
    """The branches the transform asked for, if any (``anchors``: steps, ``n`` branches each): the trial's samples, plus
    those of every branch that got a reward. A subgroup step with a branch gives the trial's samples
    ``train_metadata["branch_anchors"]`` (where its advantage from that step on comes from, added_rewards)."""
    samples = result.samples if isinstance(result.samples, list) else [result.samples]
    record = next((s.metadata["prl_transform"] for s in samples
                   if isinstance((s.metadata or {}).get("prl_transform"), dict)), None)
    request = record.get("reroll") if record else None
    if not (isinstance(request, dict) and request.get("anchors")):
        return result
    from miles.rollout.base_types import GenerateFnOutput

    if getattr(input.args, "custom_reward_post_process_path", None) != ADDED_REWARDS:
        raise ValueError(f"a reroll trains with its own reward: needs --custom-reward-post-process-path {ADDED_REWARDS}")
    jobs =[(anchor, j) for anchor in request["anchors"] for j in range(int(anchor.get("n", 1)))]
    # Seeds apart from each other and from the trial's own: a fixed seed must not reproduce a sibling's continuation
    done = await asyncio.gather(*(_branch(input, pristine, rollout_id, anchor, j, 3000017 * (k + 1))
                                  for k, (anchor, j) in enumerate(jobs)))
    record["reroll"] = [outcome for outcome, _ in done]  # the request's text stays in each branch's request.json
    new: list[Any] = []
    for outcome, branch in done:
        for s in branch:
            # The trial's reward on the sample (Miles keeps one per rollout_id; the reward metric stays the trials');
            # the branch's own reward is its group member's or its subgroup's
            s.reward = samples[0].reward
            s.metadata = {**s.metadata, "reward": samples[0].metadata.get("reward"), "prl_reroll": outcome,
                          "prl_reroll_of": request["anchors"][0].get("trial_dir")}
            s.train_metadata = {**(getattr(s, "train_metadata", None) or {}), "group_reward": outcome["reward"],
                                "added_id": f"reroll:{outcome['step']}:{outcome['branch']}",
                                "branch": {"anchor": outcome["step"], "credit": outcome["credit"]}}
        new += branch
    anchors = [{"step": a["step"], "start": a["token_start"], "credit": "subgroup"} for a in request["anchors"]
               if a.get("credit") == "subgroup"
               and any(o["status"] == "ok" and o["step"] == a["step"] for o, _ in done)]
    if anchors:
        for s in samples:  # a rerolled trial is one sample (core refuses one split by a summary)
            s.train_metadata = {**(getattr(s, "train_metadata", None) or {}), "branch_anchors": anchors}
    return GenerateFnOutput(samples=samples + new)


async def _branch(input: GenerateFnInput, pristine: Any, rollout_id: int, anchor: dict[str, Any], j: int,
                  seed_offset: int) -> tuple[dict[str, Any], list[Any]]:
    """Branch ``j`` of ``anchor``: a fresh trial of the request that replays the turns before the step
    (prl/rollout/reroll_agent.py), then the policy; its outcome and its samples (none when it failed: the trial
    trains without it)."""
    from miles.rollout.generate_hub.agentic_tool_call import generate as miles_generate
    from miles.utils.types import Sample

    from prl.rollout.harbor_training import _mismatch_clean, _shifted  # imports this module

    outcome: dict[str, Any] = {"step": anchor["step"], "branch": j, "credit": anchor.get("credit", "member"),
                               "rewritten_steps": anchor.get("rewritten_steps", []), "status": "failed"}
    t0 = time.time()
    new: list[Any] = []
    trial_dir = anchor.get("trial_dir")
    folder = Path(trial_dir) / "reroll" / f"s{anchor['step']}_b{j}" if trial_dir else None
    timeout = float(os.getenv("PRL_REROLL_TIMEOUT", "0")) or None
    try:
        if folder is None:
            raise ValueError("the trial has no directory")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "request.json").write_text(json.dumps(anchor, ensure_ascii=False))
        sample = deepcopy(pristine)
        sample.metadata = {**sample.metadata, "prl_reroll_file": str(folder / "request.json")}
        slots = _reroll_slots.setdefault(asyncio.get_running_loop(),
                                         asyncio.Semaphore(int(os.getenv("PRL_REROLL_CONCURRENCY", "32"))))
        # An aborted branch is an infrastructure failure (Harbor raised, e.g. tmux could not be installed; or Miles'
        # collect from the session server lost its connection), not a score: like an aborted trial
        # (_retry_aborted) it runs again, up to PRL_ABORT_MAX_RETRIES times, then the trial trains without it.
        attempts = 1 + int(os.getenv("PRL_ABORT_MAX_RETRIES", "2"))
        for n in range(attempts):
            async with slots:
                out = await asyncio.wait_for(miles_generate(_shifted(input, sample, seed_offset + n * 2000003)),
                                             timeout)
            new = out.samples if isinstance(out.samples, list) else [out.samples]
            outcome["attempts"] = n + 1
            if not any(s.status == Sample.Status.ABORTED for s in new):
                break
            logger.warning("[prl] reroll of rollout=%s step %s branch %s: attempt %d/%d aborted (Harbor or the "
                           "session collect failed: the warning above says which)", rollout_id, anchor["step"], j,
                           n + 1, attempts)
        else:
            raise RuntimeError(f"aborted {attempts} times (infrastructure: Harbor or the session collect)")
        outcome["trial_dir"] = next((s.metadata.get("trial_dir") for s in new if s.metadata.get("trial_dir")), None)
        reward = next((s.metadata.get("reward") for s in new if (s.metadata or {}).get("reward") is not None), None)
        if reward is None:
            raise RuntimeError(f"no verifier reward ({next((s.metadata.get('exit_status') for s in new), None)})")
        # The trials' own check (_generate_checked), without its retries: a branch is only an extra sample
        if sample.metadata.get("prl_reject_mismatch") and not _mismatch_clean(new):
            raise RuntimeError("TITO mismatch (PRL_REJECT_MISMATCH)")
        new = [s for s in new if any(s.loss_mask)]
        if not new:
            raise RuntimeError("no trained token")
        replay = folder / "reroll_replay.json"
        # The counts only: the outcome rides on every sample's metadata; the texts that differed stay in the file
        compared = {k: v for k, v in json.loads(replay.read_text()).items() if k != "differ_detail"} \
            if replay.exists() else None
        outcome.update(status="ok", reward=float(reward), samples=len(new), trained_tokens=sum(sum(s.loss_mask) for s in new), replay=compared)
    except Exception as exc:  # the trial trains without this branch
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            exc = TimeoutError(f"no result after {timeout:.0f} s (PRL_REROLL_TIMEOUT)")
        logger.warning("[prl] reroll of rollout=%s step %s branch %s failed: %s", rollout_id, anchor["step"], j,
                       exc, exc_info=True)
        outcome["reason"] = f"{type(exc).__name__}: {exc}"[:1000]
        new = []
    outcome["seconds"] = round(time.time() - t0, 2)
    if folder is not None:
        try:
            (folder / "outcome.json").write_text(json.dumps(outcome) + "\n")
        except OSError:
            logger.exception("[prl] reroll: could not write to %s", folder)
    logger.info("[prl] reroll rollout=%s step %s branch %s: %s in %.0f s, reward %s", rollout_id, anchor["step"],
                j, outcome["status"], outcome["seconds"], outcome.get("reward"))
    return outcome, new
