"""Training adapter around Miles' generic agentic generator and Harbor wrapper.

Use --custom-generate-function-path prl.rollout.harbor_training.generate
and --custom-agent-function-path prl.rollout.harbor_training.run with the
training_samples picker/postprocessor and TITO v2. This preserves Miles tracing,
sampling and compact-rollout handling. Per-request evaluation state travels via
sample -> agent -> session metadata (MODE alone cannot distinguish periodic eval).

Harbor/infrastructure errors still raise in harbor_agent; this boundary converts
them to explicit aborted metadata. Unscored training trials are also aborted: no fabricated zero
reward is learned. Existing verifier reward=0 remains a valid failed answer. Policy limits (agent timeout, length) do
not raise in harbor_agent, but a training trial they end before the verifier ran has no eval_report and is aborted
like an infrastructure failure: resampled, not scored 0 (also a shell the model closed). Evaluation scores such a
trial 0, and by default any trial that failed, as Harbor and TB-dev's own scoring do. With PRL_EVAL_SANDBOX_RETRIES
a failure of the sandbox service (SANDBOX_FAILURES: a rate limit, an internal error of the service, a command whose end
event was lost, a request to the service that timed out or lost its connection) is run again instead: it says nothing about the policy, and as 0 it makes a score depend on the
account's sandbox quota. An aborted training trial is first run again on its own, in a fresh session (PRL_ABORT_MAX_RETRIES); only
when every attempt aborts does the aborted sample reach Miles, whose pre-put filter (check_no_aborted) then discards
the whole prompt group; one new prompt replaces it (rollout/refill.py). Refilling cost a full extra trial wave at the
end of the rollout: in one run, 24 of 68 updates refilled a group and their rollouts took 547 s against 373 s.
Configuration:
    PRL_ABORT_MAX_RETRIES: extra attempts of an aborted training trial (same prompt, fresh session, shifted
        seed), default 2; 0 restores the whole-group refill on the first abort.
    PRL_EVAL_SANDBOX_RETRIES: extra attempts of an evaluation trial that failed on the sandbox service, default 0
        (it scores 0, comparable with every earlier evaluation); scripts/tools/tbdev_score.py --samples then drops the
        failed attempts from the score.
        Kept samples carry metadata["prl_abort_retries"] (rollout/abort_retried_trials). An aborted reroll branch
        is run again as often (its outcome's "attempts").
    PRL_REJECT_MISMATCH: 0 (default) restores previous behavior; 1 rejects
        the entire trial if any raw TITO leaf mismatches or lacks its check.
        Training only; periodic eval remains unfiltered. A reroll that fails the same check is dropped, not retried
        (its trial trains without it).
    PRL_MISMATCH_MAX_RETRIES: extra same-prompt attempts, default 2.
        Exhaustion returns ABORTED/reward=None; Miles discards/refills the group.
Requires synchronous training (ASYNC=0); retries finish before this update and
use fresh Miles sessions. The trajectory transform (PRL_TRANSFORM, e.g. atlas; prl/transforms/core.py)
runs here, and only here, on each finished training trial. Summary configuration remains in HARBOR_ENV_KWARGS.
A transform's ``step.reroll(n, credit)`` branches are run right after the transform (prl/rollout/branching.py).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput

from prl.rollout.branching import ADDED_REWARDS, _reroll  # noqa: F401 -- ADDED_REWARDS re-exported

logger = logging.getLogger(__name__)

# Failures of the sandbox service, not of the policy or the task: the exception type or text Harbor reports
# (prl/rollout/harbor_agent.py raises with both). scripts/tools/tbdev_score.py keeps the same list.
SANDBOX_FAILURES = ("RateLimitException", "SandboxException", "Command ended without an end event",
                    # requests to the sandbox service that timed out or lost their connection (the e2b SDK's errors)
                    "ReadTimeout", "RemoteProtocolError", "TimeoutException")
# What made the attempt in flight abort, set by run() for _retry_aborted (the agent function runs inside the attempt's
# await, so it sees the attempt's dict; the aborted sample Miles returns carries no trace of the agent's exception)
_failure: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("failure", default=None)


async def run(*args: Any, metadata: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    from prl.rollout.harbor_agent import run as run_harbor

    flags = {"prl_training_contract": True, "prl_evaluation": metadata["prl_evaluation"],
             "prl_reject_mismatch": metadata.get("prl_reject_mismatch", False)}
    try:
        result = await run_harbor(*args, metadata=metadata, **kwargs)
    except Exception as exc:
        logger.exception("[prl] Harbor failed: aborting trial instead of assigning reward zero")
        note = _failure.get()
        if note is not None:
            note["sandbox"] = any(failure in str(exc) for failure in SANDBOX_FAILURES)
        # The async producer replaces an aborted trial at once. Four jobs on one sandbox key (refill 5/s) kept each other
        # at 429 for an hour (15k failed trials per job and no update): hold the slot, so the load falls.
        if "RateLimitException" in str(exc):
            await asyncio.sleep(random.uniform(0.5, 1.5) * float(os.getenv("PRL_RATE_LIMIT_WAIT", "60")))
        return {**flags, "prl_aborted": True, "reward": None}
    if not flags["prl_evaluation"] and not result.get("eval_report"):
        logger.warning("[prl] Unscored training trial aborted: %s", result.get("exit_status"))
        return {**flags, "prl_aborted": True, "reward": None}
    return {**result, **flags, "prl_aborted": False}


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    from miles.rollout.generate_hub.agentic_tool_call import generate as miles_generate

    if input.args.use_session_server != "v2":
        raise ValueError("Harbor training adapter requires TITO v2")
    enabled = os.getenv("PRL_REJECT_MISMATCH", "0")
    if enabled not in ("0", "1"):
        raise ValueError("PRL_REJECT_MISMATCH must be 0 or 1")
    reject = enabled == "1" and not input.evaluation
    input.sample.metadata = {
        **input.sample.metadata,
        "prl_training_contract": True,
        "prl_evaluation": input.evaluation,
        "prl_reject_mismatch": reject,
    }
    # Even single-leaf trials belong to the compact rollout schedule when mixed
    # with summarized trials. Keep IDs consistent for normalization and batching.
    rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
    if rollout_id is None:
        raise ValueError("Training sample needs a rollout_id or index")
    if reject:
        if os.getenv("ASYNC", "0") != "0" or getattr(input.args, "fully_async", False):
            raise ValueError("Mismatch refill requires synchronous training")
        retries = int(os.getenv("PRL_MISMATCH_MAX_RETRIES", "2"))
        if retries < 0:
            raise ValueError("PRL_MISMATCH_MAX_RETRIES must be nonnegative")

        async def attempt(attempt_input: GenerateFnInput) -> GenerateFnOutput:
            return await _generate_checked(attempt_input, miles_generate, retries, rollout_id)
    else:
        attempt = miles_generate
    pristine = None if input.evaluation else deepcopy(input.sample)  # a reroll starts from the request as it came
    knob, default = ("PRL_EVAL_SANDBOX_RETRIES", "0") if input.evaluation else ("PRL_ABORT_MAX_RETRIES", "2")
    abort_retries = int(os.getenv(knob, default))
    if abort_retries < 0:
        raise ValueError(f"{knob} must be nonnegative")
    result = await _retry_aborted(input, attempt, abort_retries, rollout_id, sandbox_only=input.evaluation)
    if not input.evaluation:
        result = await _transform(input, result)
        result = await _reroll(input, result, pristine, rollout_id)
    for sample in result.samples:
        sample.rollout_id = rollout_id
    return result


def _shifted(input: GenerateFnInput, sample: Any, offset: int) -> GenerateFnInput:
    """The same request with a copy of ``sample`` and the sampling seed moved by ``offset`` (a fixed seed must not
    reproduce exactly the rejected trajectory)."""
    sampling_params = deepcopy(input.sampling_params)
    for key in ("sampling_seed", "seed"):
        if sampling_params.get(key) is not None:
            sampling_params[key] = (int(sampling_params[key]) + offset) % (2**31)
    return replace(input, sample=deepcopy(sample), sampling_params=sampling_params)


async def _retry_aborted(input: GenerateFnInput, attempt: Any, retries: int, rollout_id: int,
                         sandbox_only: bool = False) -> GenerateFnOutput:
    """Run the trial; while it aborts (an infrastructure failure), run it again on its own, up to ``retries`` times,
    so its group stays whole instead of being discarded and refilled by Miles. ``sandbox_only`` (evaluation): only
    while the abort is a failure of the sandbox service (SANDBOX_FAILURES); any other abort scores 0 at once."""
    from miles.utils.types import Sample

    pristine = deepcopy(input.sample) if retries else None  # Miles' generator writes into the sample it gets
    for n in range(retries + 1):
        # Abort retries shift the seed by multiples of 2000003; mismatch retries inside use multiples of 1000003
        note: dict[str, Any] = {}
        token = _failure.set(note)
        try:
            result = await attempt(input if n == 0 else _shifted(input, pristine, n * 2000003))
        finally:
            _failure.reset(token)
        if not any(s.status == Sample.Status.ABORTED for s in result.samples):
            if retries:
                for sample in result.samples:
                    sample.metadata["prl_abort_retries"] = n
            if n:
                logger.info("[prl] aborted trial accepted on retry rollout=%s retries=%d", rollout_id, n)
            return result
        if any(s.metadata.get("prl_mismatch_exhausted") for s in result.samples):
            return result  # PRL_MISMATCH_MAX_RETRIES already spent its own attempts on this trial
        if sandbox_only and not note.get("sandbox"):
            return result
        if n < retries:
            logger.warning("[prl] trial aborted, retrying it alone rollout=%s attempt=%d/%d",
                           rollout_id, n + 1, retries)
    if retries:
        logger.warning("[prl] trial aborted %d times rollout=%s; %s", retries + 1, rollout_id,
                       "it scores 0" if sandbox_only else "Miles discards and refills its group")
    return result


async def _transform(input: GenerateFnInput, result: GenerateFnOutput) -> GenerateFnOutput:
    """The trajectory transform chain (e.g. atlas: a judge model labels the steps) on a finished training trial."""
    from prl.transforms import core

    # Checked before importing Miles: without a transform (the default) this is a no-op, also outside the container
    if not core.configured():
        return result
    from miles.rollout.base_types import GenerateFnOutput
    from miles.utils.types import Sample

    samples = result.samples if isinstance(result.samples, list) else [result.samples]
    if any(s.status == Sample.Status.ABORTED for s in samples):
        return result
    return GenerateFnOutput(samples=await core.run(samples, input.state.tokenizer, input.args))


def _mismatch_clean(samples: list[Any]) -> bool:
    """Every TITO leaf of the trial was checked against the session and none mismatched (PRL_REJECT_MISMATCH)."""
    checks = [s.metadata.get("prl_mismatch_checks") for s in samples]
    return bool(checks) and all(
        isinstance(c, dict) and c.get("checked_leaves", 0) > 0 and c.get("rejected") == [] for c in checks
    ) and all(s.metadata.get("tito_session_mismatch") == [] for s in samples)


async def _generate_checked(input: GenerateFnInput, generator: Any, retries: int, rollout_id: int) -> GenerateFnOutput:
    from miles.rollout.base_types import GenerateFnOutput
    from miles.utils.types import Sample

    rejected_trials: list[str] = []
    for attempt in range(retries + 1):
        attempt_input = _shifted(input, input.sample, attempt * 1000003)
        result = await generator(attempt_input)  # creates and cleans up a fresh TITO session
        if any(s.status == Sample.Status.ABORTED for s in result.samples):
            return result  # Infrastructure abort retains the existing group-refill behavior.
        if _mismatch_clean(result.samples):
            for sample in result.samples:
                sample.metadata["prl_mismatch_retries"] = attempt
                sample.metadata["prl_rejected_trials"] = rejected_trials.copy()
            logger.info("[prl] mismatch_refill accepted rollout=%s retries=%d", rollout_id, attempt)
            return result
        trial_dir = next((s.metadata.get("trial_dir") for s in result.samples if s.metadata.get("trial_dir")), None)
        if trial_dir:
            rejected_trials.append(str(trial_dir))
            # Harbor keeps the full agent trajectory. Preserve the collected
            # training spans too, before dropping them from reward normalization.
            evidence = {"rollout_id": rollout_id, "attempt": attempt, "max_retries": retries,
                        "samples": [{"tokens": s.tokens, "loss_mask": s.loss_mask,
                                     "response_length": s.response_length, "reward": s.reward,
                                     "leaf": s.metadata.get("leaf"),
                                     "mismatch": s.metadata.get("tito_session_mismatch"),
                                     "checks": s.metadata.get("prl_mismatch_checks")}
                                    for s in result.samples]}
            (Path(trial_dir) / "training_mismatch_rejection.json").write_text(json.dumps(evidence) + "\n")
        logger.warning("[prl] mismatch_refill rejected rollout=%s attempt=%d/%d trial=%s",
                       rollout_id, attempt, retries, trial_dir)
    sample = deepcopy(input.sample)
    sample.status = Sample.Status.ABORTED
    sample.reward = None
    sample.metadata.update(prl_mismatch_exhausted=True, prl_rejected_trials=rejected_trials)
    logger.warning("[prl] mismatch_refill exhausted rollout=%s; discard/refill group", rollout_id)
    return GenerateFnOutput(samples=[sample])


def _add_arguments(parser: Any) -> None:
    from miles.rollout.generate_hub.agentic_tool_call import generate as miles_generate

    miles_generate.add_arguments(parser)


generate.add_arguments = _add_arguments
