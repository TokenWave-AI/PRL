"""scripts/tools/tbdev_score.py: unscored trials count 0, and --samples drops the attempts an evaluation ran again after
a failure of the sandbox service, so a task scores the N trials it asked for."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from prl.rollout.harbor_training import SANDBOX_FAILURES

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "tools" / "tbdev_score.py"


def _trial(root: Path, task: str, n: int, reward: float | None, exception: tuple[str, str] | None = None) -> None:
    trial = root / f"{task}__2026111{n}-000000__x{n}"
    trial.mkdir(parents=True)
    result = {"task_name": task, "trial_name": trial.name, "started_at": f"2026-11-12T00:00:0{n}Z",
              "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
              "exception_info": None if exception is None else {"exception_type": exception[0],
                                                                "exception_message": exception[1]}}
    (trial / "result.json").write_text(json.dumps(result))


def _score(*args: str) -> str:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, check=True).stdout


def test_retried_sandbox_failures_drop_out_with_samples(tmp_path: Path) -> None:
    quota = ("RateLimitException", "429: Rate limit exceeded - account resource quota exhausted")
    _trial(tmp_path, "a", 0, None, quota)  # failed, then run again: 3 trials for 2 samples
    _trial(tmp_path, "a", 1, 1.0)
    _trial(tmp_path, "a", 2, 0.0)
    _trial(tmp_path, "b", 0, None, quota)  # failed on every attempt: still one of b's 2 trials, scored 0
    _trial(tmp_path, "b", 1, 1.0)
    _trial(tmp_path, "c", 0, None, ("RuntimeError", "failed to send non-blocking keys"))  # not the sandbox service
    _trial(tmp_path, "c", 1, 1.0)
    plain = _score(str(tmp_path))
    assert "trials=7 " in plain and "mean=0.4286" in plain
    kept = _score("--samples", "2", str(tmp_path))
    assert "trials=6 " in kept and "trials_per_task={2: 3}" in kept and "dropped_sandbox_failures=1" in kept
    assert "mean=0.5000" in kept


def test_dropping_sandbox_failures_scores_the_mean_of_task_means(tmp_path: Path) -> None:
    lost = ("ReadTimeout", "timed out")  # a request to the sandbox service (e2b SDK)
    _trial(tmp_path, "a", 0, None, lost)
    _trial(tmp_path, "a", 1, 1.0)
    _trial(tmp_path, "b", 0, 0.0)
    _trial(tmp_path, "b", 1, 1.0)
    _trial(tmp_path, "c", 0, None, ("RuntimeError", "failed to send non-blocking keys"))  # scores 0, stays
    _trial(tmp_path, "c", 1, 1.0)
    out = _score("--drop-sandbox-failures", str(tmp_path))
    assert "trials=5 " in out and "dropped_sandbox_failures=1" in out
    assert "mean=0.6667" in out  # (1 + 0.5 + 0.5) / 3; the mean over the 5 trials would be 0.6


def test_the_scorer_knows_the_same_sandbox_failures() -> None:
    tree = ast.parse(TOOL.read_text())
    value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and getattr(node.targets[0], "id", None) == "SANDBOX_FAILURES")
    assert ast.literal_eval(value) == SANDBOX_FAILURES
