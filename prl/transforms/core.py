"""Trajectory transforms: the one user function between a finished training trial and the trainer.

Where it runs. ``prl.rollout.harbor_training.generate`` (Miles --custom-generate-function-path) calls ``run`` on
each finished, non-aborted training trial (never on evaluation trials), after Miles' session postprocessing
(``prl.rollout.training_samples``: ownership masks, summary calls masked). The result goes to Miles' GRPO
advantage, then ``prl.algo.step_advantage`` (step weights), then the loss.

What a transform is and what it may do to a trial: prl/transforms/api.py (Trajectory, Step). The text
edits are re-encoded by prl/transforms/tokens.py; ask (the run's model) is prl/transforms/ask.py.
This module runs the selected transforms on a trial and turns their decisions into the samples Miles trains.

After each function the framework writes the step decisions into the samples, checks them (Miles' own
``Sample.validate``, a 0/1 loss mask, one reward for the trial's samples (unchanged unless ``traj.reward`` was set),
rollout_id and prompt unchanged, at least one trained token) and drops samples left without a trained token. A
function that raises, returns an invalid result, skips or runs out of time leaves the trial as it was; after PRL_TRANSFORM_MAX_FAILURES such trials in a row the run stops, so a
systematic failure cannot silently train plain GRPO.

Configuration (env; scripts/run.sh forwards PRL_TRANSFORM*):
    PRL_TRANSFORM       a transform name, or a chain applied in order, separated by "," or "+" (e.g.
                                   ``rewrite,atlas``; submit.sh's TRANSFORM takes ``rewrite+atlas``);
                                   ``none`` or unset: no transform (default)
    PRL_TRANSFORM_ARGS  JSON object overriding the function's default params; for a chain, keyed by name
    PRL_TRANSFORM_MAX_FAILURES  trials in a row left unchanged by a failure (error, invalid result, skip,
                                   timeout) before the run stops (default 32; 0 disables). Trials a transform leaves
                                   as they are on purpose neither count nor reset the count
    PRL_TRANSFORM_TIMEOUT  seconds one transform may take on one trial (default 1800; 0: no limit); after
                                   it the trial trains unchanged (error). The pre-run check uses the same limit

Records: every kept sample of the trial gets ``metadata["prl_transform"]`` and the trial dir ``transform.json``:
per transform its status (``ok`` / ``skipped`` / ``error``), reason, seconds, ``traj.record``, the steps it changed
(``rewritten_steps`` / ``deleted_steps``, ``masked_steps``, ``weighted_steps`` {step: new weight},
``token_weighted_steps`` (weight_text), ``reroll_steps``) and the reward before and after; trained tokens before and
after; whether tokens changed; the samples dropped (left without a trained token).
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import inspect
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prl.transforms.api import CREDITS, MAX_BRANCHES, Step, Trajectory  # noqa: F401 -- re-exported
from prl.transforms.ask import Unanswered, _calls, ask  # noqa: F401 -- re-exported
from prl.transforms.tokens import _runs, _splice  # noqa: F401 -- re-exported (tests)

logger = logging.getLogger(__name__)

# The repo-root package that holds the transforms, one file per name (on PYTHONPATH in the Miles container)
PACKAGE = "transforms"


def _fingerprint(samples: list[Any]) -> list[tuple[Any, ...]]:
    """What trains: tokens, mask, reward, step and token weights."""
    return [(tuple(s.tokens), tuple(s.loss_mask), _reward(s),
             json.dumps({k: (getattr(s, "train_metadata", None) or {}).get(k)
                         for k in ("step_weights", "token_weight_spans")})) for s in samples]


def _spec() -> tuple[str, str, str]:
    spec = os.getenv("PRL_TRANSFORM", "").strip() or "none"
    return spec, os.getenv("PRL_TRANSFORM_ARGS", "").strip(), "PRL_TRANSFORM"


def available() -> list[str]:
    spec = importlib.util.find_spec(PACKAGE)
    folders = list(spec.submodule_search_locations or []) if spec else []
    return sorted(p.stem for folder in folders for p in Path(folder).glob("*.py") if not p.stem.startswith("_"))


def _resolve(name: str) -> Callable[..., Any]:
    if not name.isidentifier():
        raise ValueError(f"transform name {name!r}: use the file name of {PACKAGE}/<name>.py")
    try:
        module = importlib.import_module(f"{PACKAGE}.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{PACKAGE}.{name}":
            raise  # the transform's own imports
        raise ValueError(f"no transform {name!r}: {PACKAGE}/{name}.py not found; available: {available()}") from None
    fn = getattr(module, name, None)
    if not callable(fn):
        raise ValueError(f"{PACKAGE}/{name}.py must define a function {name}(traj, **params)")  # noqa: TRY004
    return fn


def effective_params(fn: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
    """The params ``fn`` runs with: its keyword defaults, then ``params`` (recorded in transform.json)."""
    defaults = {name: p.default for name, p in list(inspect.signature(fn).parameters.items())[1:]
                if p.default is not inspect.Parameter.empty}
    return {**defaults, **params}


def configured() -> list[tuple[str, Callable[..., Any], dict[str, Any]]]:
    """The selected chain as (item, function, params); [] without a transform. Raises on any configuration error."""
    spec, raw, source = _spec()
    # "+" as well as ",": a job's variables reach it through pjsub -x / qsub -v, which split on commas
    items = [x.strip() for x in re.split(r"[,+]", spec) if x.strip() not in ("", "none")]
    if not items:
        return []
    params = json.loads(raw or "{}")
    if not isinstance(params, dict):
        raise ValueError(f"{source}_ARGS must be a JSON object")  # noqa: TRY004 -- a config value
    if len(items) > 1:
        if set(params) - set(items):
            raise ValueError(f"{source}_ARGS of a chain is keyed by its items {items}; got {sorted(params)}")
        per_item = [params.get(item, {}) for item in items]
    else:
        per_item = [params]
    chain = []
    for item, p in zip(items, per_item):
        fn = _resolve(item)
        if not isinstance(p, dict):
            raise ValueError(f"params of {item!r} must be a JSON object")  # noqa: TRY004 -- a config value
        try:
            inspect.signature(fn).bind(None, **p)
        except TypeError as exc:
            raise ValueError(f"params {p} do not fit {item!r}{inspect.signature(fn)}: {exc}") from None
        chain.append((item, fn, p))
    return chain


# transform.json's step lists, as the log line of a trajectory and the check's line name them
CHANGES = (("weighted_steps", "weighted"), ("masked_steps", "masked"), ("token_weighted_steps", "token-weighted"),
           ("rewritten_steps", "rewritten"), ("deleted_steps", "deleted"), ("reroll_steps", "rerolled"))


def changed_steps(record: dict[str, Any]) -> dict[str, list[int]]:
    """The steps the chain changed, by kind (a step in several functions of the chain counts once per kind)."""
    out: dict[str, set[int]] = {}
    for entry in record["chain"]:
        for key, kind in CHANGES:
            out.setdefault(kind, set()).update(int(i) for i in entry.get(key) or [])
    out["rewritten"] -= out["deleted"]  # a deleted step is in rewritten_steps too
    return {kind: sorted(steps) for kind, steps in out.items() if steps}


def describe_changes(changed: dict[str, list[int]]) -> str:
    """The counts of changed_steps, e.g. "5 weighted, 2 masked", or "none"."""
    return ", ".join(f"{len(steps)} {kind}" for kind, steps in changed.items()) or "none"


def _trained(samples: list[Any]) -> int:
    return sum(sum(s.loss_mask) for s in samples)


def _prompt(sample: Any) -> tuple[int, ...]:
    return tuple(sample.tokens[: len(sample.tokens) - sample.response_length])


def _reward(sample: Any) -> str:
    # Miles' reward may be a dict (reward_key); compare its JSON
    return json.dumps(getattr(sample, "reward", None), sort_keys=True, default=str)


def _check(before: list[Any], after: list[Any], tokens_changed: bool, args: Any, reward_changed: bool = False) -> None:
    if len(after) > len(before):
        raise ValueError(f"{len(after)} samples from {len(before)}: a transform may drop samples, not add them")
    ids = {getattr(s, "rollout_id", None) for s in before}
    prompts = {_prompt(s) for s in before}
    rewards = {_reward(s) for s in after}
    # Miles requires one reward per rollout (train_data_conversion._normalize_rewards_by_rollout)
    if len(rewards) > 1 or (not reward_changed and not rewards <= {_reward(s) for s in before}):
        raise ValueError(f"the trial's samples must share its reward; got {sorted(rewards)}")
    for s in after:
        if getattr(s, "rollout_id", None) not in ids:
            raise ValueError(f"rollout_id changed to {getattr(s, 'rollout_id', None)}; the trial had {sorted(ids)}")
        if len(s.tokens) < s.response_length or len(s.loss_mask) != s.response_length:
            raise ValueError(f"{len(s.tokens)} tokens, response_length {s.response_length}, "
                             f"{len(s.loss_mask)} loss_mask entries")
        if getattr(s, "rollout_log_probs", None) is not None and len(s.rollout_log_probs) != s.response_length:
            raise ValueError(f"{len(s.rollout_log_probs)} rollout_log_probs, response_length {s.response_length}")
        if hasattr(s, "validate"):
            s.validate()  # Miles' Sample: also weight-version spans and the other per-token fields
        if not set(s.loss_mask) <= {0, 1}:
            raise ValueError(f"loss_mask values {sorted(set(s.loss_mask))}; only 0 and 1")
        weights = (s.train_metadata or {}).get("step_weights")
        runs = sum(1 for trained, _, _ in _runs(list(s.loss_mask)) if trained)
        if weights and len(weights) != runs:
            raise ValueError(f"{len(weights)} step_weights for {runs} trained runs (step_advantage would misalign them)")
        for span in (s.train_metadata or {}).get("token_weight_spans") or []:
            t0, t1, w = span
            if not (isinstance(t0, int) and isinstance(t1, int) and 0 <= t0 < t1 <= s.response_length
                    and isinstance(w, (int, float)) and math.isfinite(w) and all(s.loss_mask[t0:t1])):
                raise ValueError(f"token weight span {span}: needs 0 <= start < end <= {s.response_length}, a finite "
                                 "weight, and trained tokens only")
        if _prompt(s) not in prompts:
            raise ValueError("the prompt tokens changed; a transform may change the response only")
    if tokens_changed and args is not None and (getattr(args, "use_tis", False)
                                                or getattr(args, "use_rollout_logprobs", False)):
        raise ValueError("tokens changed but --use-tis/--use-rollout-logprobs is on: rewritten tokens have no rollout "
                         "log probs")
    if _trained(after) == 0:
        raise ValueError("no trained token left in the trial")


_failures = 0


def _count(status: list[str], changed: bool, trial_dir: str | None) -> None:
    global _failures
    if all(s == "ok" for s in status):
        # Only a trial the transform changed resets the count: one it leaves as it is on purpose (rewrite on a failed
        # trial) says nothing about its model being reachable, and would otherwise hide an outage
        if changed:
            _failures = 0
        return
    _failures += 1
    limit = int(os.getenv("PRL_TRANSFORM_MAX_FAILURES", "32"))
    if limit and _failures >= limit:
        raise RuntimeError(f"[prl] transform: {_failures} trials in a row left unchanged (last {trial_dir}: its "
                           f"transform.json and transform_calls.json say why, e.g. a model that stayed unavailable); "
                           f"stopping instead of training them untransformed (PRL_TRANSFORM_MAX_FAILURES)")


async def run(samples: list[Any], tokenizer: Any, args: Any = None, strict: bool = False) -> list[Any]:
    """The configured chain on one training trial's samples; returns the samples to train on. ``strict`` (the pre-run
    check, prl/transforms/check.py): a failing function raises instead of leaving the trial unchanged."""
    chain = configured()
    if not chain:
        return samples
    before = _trained(samples)
    start = _fingerprint(samples)
    log: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    tokens_changed = False
    new_tokens = 0
    trial_dir = None
    reroll: dict[str, Any] | None = None  # the chain's reroll request (Trajectory._reroll_request), one transform's
    timeout = float(os.getenv("PRL_TRANSFORM_TIMEOUT", "1800")) or None
    for item, fn, params in chain:
        original = copy.deepcopy(samples)
        entry: dict[str, Any] = {"transform": item, "params": effective_params(fn, params)}
        t0 = time.time()
        traj = None
        mine: list[dict[str, Any]] = []
        token = _calls.set(mine)
        try:
            traj = Trajectory.of(samples, tokenizer)
            trial_dir = traj.trial_dir
            call = fn(traj, **params) if inspect.iscoroutinefunction(fn) else asyncio.to_thread(fn, traj, **params)
            # Synchronous training waits for its slowest trial; a hung model call must not hold the whole step
            out = await asyncio.wait_for(call, timeout)
            if out is not None and out is not traj:
                raise TypeError(f"returned {type(out).__name__}; return the trajectory it was given, or None")
            entry["record"] = traj.record
            if traj.skipped is not None:
                samples = original
                entry.update(status="skipped", reason=str(traj.skipped))
            else:
                originals = {tuple(s.tokens) for s in original}
                if any(tuple(s.tokens) not in originals for s in traj.samples) and traj._edited():
                    raise ValueError("changed sample tokens directly and used the step API in one call; use one")
                traj._apply()
                changed = any(tuple(s.tokens) not in originals for s in traj.samples)
                reward_changed = traj.reward != traj._initial_reward
                _check(original, traj.samples, changed, args, reward_changed)
                if traj._extras:
                    if any(tuple(s.tokens) not in originals for s in traj._extras):
                        changed = True
                    _check(original, traj._extras, changed, args, reward_changed)
                samples = traj.samples + traj._extras
                tokens_changed |= changed
                new_tokens += traj._new_tokens
                if reward_changed:
                    entry["reward"] = [traj._initial_reward, traj.reward]
                if traj._reroll is not None:
                    steps = [a["step"] for a in traj._reroll["anchors"]]
                    if reroll is not None:
                        raise ValueError(f"steps {steps} rerolled after an earlier transform rerolled steps "
                                         f"{[a['step'] for a in reroll['anchors']]}: one transform of the chain rerolls")
                    entry["reroll_steps"] = steps
                    reroll = traj._reroll
                rewritten = [s.index for s in traj.steps if s._rewritten()]
                if rewritten:
                    entry["rewritten_steps"] = rewritten
                    entry["deleted_steps"] = [s.index for s in traj.steps if s.deleted]
                # what this function decided, against the steps as it got them (_state: weight, masked, ...)
                live = [(s, was) for s, was in zip(traj.steps, traj._initial) if not s.deleted]
                if masked := [s.index for s, was in live if s.masked and not was[1]]:
                    entry["masked_steps"] = masked
                if weighted := {str(s.index): float(s.weight) for s, was in live if not s.masked and s.weight != was[0]}:
                    entry["weighted_steps"] = weighted
                if spanned := [s.index for s, _ in live if s._spans]:
                    entry["token_weighted_steps"] = spanned
                entry["status"] = "ok"
        except Unanswered as exc:  # the model gave nothing usable: as traj.skip
            samples = original
            entry.update(status="skipped", reason=str(exc), record=traj.record if traj is not None else {})
        except Exception as exc:  # any failure of the user function: the trial trains unchanged
            if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) and not str(exc):  # distinct before Python 3.11
                late = TimeoutError(f"{item}: no result after {timeout:.0f} s (PRL_TRANSFORM_TIMEOUT); a synchronous "
                                    "function keeps its thread until it returns, an async one is cancelled")
                if strict:
                    raise late from exc
                exc = late
            if strict:
                raise
            logger.exception("[prl] transform %s failed on %s; trial kept unchanged", item, trial_dir)
            samples = original
            entry.update(status="error", reason=f"{type(exc).__name__}: {exc}"[:2000])
        finally:
            _calls.reset(token)
        if mine:
            usage = [c.get("usage_total") or {} for c in mine]
            entry["calls"] = {"n": len(mine), "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
                              "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage)}
            calls += [{"transform": item, **c} for c in mine]
        entry["seconds"] = round(time.time() - t0, 2)
        log.append(entry)
    kept = [s for s in samples if any(s.loss_mask)]
    changed = _fingerprint(kept) != start or reroll is not None
    record = {"chain": log, "trained_tokens_before": before, "trained_tokens_after": _trained(kept),
              "tokens_changed": tokens_changed, "new_tokens": new_tokens if tokens_changed else 0, "changed": changed,
              "dropped_samples": len(samples) - len(kept)}
    if reroll is not None:
        record["reroll"] = reroll  # harbor_training runs it and replaces it with its outcome
    samples = kept
    for s in samples:
        # Miles ships train_metadata only when the batch's first sample has it (train_data_conversion.py)
        s.train_metadata = s.train_metadata if s.train_metadata is not None else {}
        s.metadata["prl_transform"] = record
    if trial_dir:
        try:
            (Path(trial_dir) / "transform.json").write_text(json.dumps(record, default=str) + "\n")
            if calls:
                (Path(trial_dir) / "transform_calls.json").write_text(
                    json.dumps(calls, default=str, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("[prl] transform: could not write to %s", trial_dir)
    logger.info("[prl] transform %s on %s: %s; steps changed: %s; %d/%d trained tokens kept",
                ",".join(e["transform"] for e in log), trial_dir, ",".join(e["status"] for e in log),
                describe_changes(changed_steps(record)), record["trained_tokens_after"], before)
    if not strict:
        _count([e["status"] for e in log], changed, trial_dir)
    return samples
