"""Execute a trial's keystrokes in a fresh sandbox of its task and run its verifier (Harbor Trial, prl/rollout/replay_agent.py).

    reward, outputs = await replay(trial_dir, turns)

``trial_dir``: a finished Harbor trial, whose config.json gives the task, the sandbox and the agent's settings. ``turns``: for each
turn, its list of commands ({"keystrokes", "duration"}). ``outputs``: for each turn that ran, {"timed_out", "output"}.
As a generated trial (prl/rollout/harbor_training.py): a verifier's reward counts even after an agent timeout; a run without one
gets no reward, it is run again (a rate limit waits first). After ATTEMPTS, ReplayFailed: no reward is ever assumed.
PRL_REPLAY_CONCURRENCY limits the sandboxes in use (default 32).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

TIMEOUT = 1800.0
ATTEMPTS = 3
_slots: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


class ReplayFailed(Exception):
    pass


async def replay(trial_dir: str, turns: list[list[dict[str, Any]]], name: str = "replay") -> tuple[float, list[dict[str, Any]]]:
    why = ""
    for attempt in range(ATTEMPTS):
        try:
            return await _once(Path(trial_dir) / name / f"attempt{attempt}", Path(trial_dir), turns)
        except ReplayFailed as exc:
            why = str(exc)
            if "RateLimitException" in why:  # as harbor_training.run: hold the slot, so the load falls
                await asyncio.sleep(random.uniform(0.5, 1.5) * float(os.getenv("PRL_RATE_LIMIT_WAIT", "60")))
    raise ReplayFailed(f"{ATTEMPTS} attempts: {why}")


async def _once(work: Path, base: Path, turns: list[list[dict[str, Any]]]) -> tuple[float, list[dict[str, Any]]]:
    request, outputs = work / "request.json", work / "outputs.json"
    slots = _slots.setdefault(asyncio.get_running_loop(),
                              asyncio.Semaphore(int(os.getenv("PRL_REPLAY_CONCURRENCY", "32"))))
    async with slots:
        try:
            from harbor.models.trial.config import TrialConfig
            from harbor.trial.trial import Trial

            work.mkdir(parents=True, exist_ok=True)
            request.write_text(json.dumps({"turns": turns, "outputs": str(outputs)}, ensure_ascii=False))
            config = TrialConfig.model_validate_json((base / "config.json").read_text())
            config.trial_name, config.trials_dir = "trial", work
            config.agent.name = None
            config.agent.import_path = "prl.rollout.replay_agent:ReplayAgent"
            config.agent.kwargs = {**config.agent.kwargs, "replay_file": str(request)}
            trial = await Trial.create(config)
            result = await asyncio.wait_for(trial.run(), timeout=TIMEOUT)
        except Exception as exc:
            raise ReplayFailed(f"{type(exc).__name__}: {exc}"[:1000]) from exc
    rewards = dict(getattr(result.verifier_result, "rewards", None) or {})
    if not rewards:
        info = result.exception_info
        raise ReplayFailed(f"no verifier reward: {info.exception_type if info else ''} {info.exception_message if info else ''}"[:1000])
    return float(rewards.get("reward", next(iter(rewards.values())))), json.loads(outputs.read_text()) if outputs.exists() else []
