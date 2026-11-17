"""scripts/checks/replay_check.py: the verdict of the chain step replay, on case directories as the step leaves them."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("replay_check", ROOT / "scripts/checks/replay_check.py")
replay_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_check)

UPDATES = {0: {"train/loss": 0.0054, "train/grad_norm": 0.351, "train/kl_loss": 0.0},
           1: {"train/loss": -0.0423, "train/grad_norm": 0.332, "train/kl_loss": 0.00035},
           2: {"train/loss": 0.0101, "train/grad_norm": 0.340, "train/kl_loss": 0.0007}}


def log(path: Path, updates: dict, scale: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"(MegatronTrainRayActor pid=1) model.py:902 - step {n}: "
             + "{" + ", ".join(f"'{k}': {v * scale if k != 'train/kl_loss' else v}" for k, v in m.items())
             + f", 'train/step': {n}" + "}" for n, m in updates.items()]
    path.write_text("\n".join(["perf 0: {'perf/train_time': 99.3}", *lines]) + "\n")


def files(case: Path, **values: str) -> None:
    case.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (case / name).write_text(value + "\n")


def healthy(tmp_path: Path) -> Path:
    """A replay root where every case behaved as it should."""
    source = tmp_path / "smoke"
    log(source / "train.log", {n: UPDATES[n] for n in (0, 1)})
    root = tmp_path / "replay"
    files(root, source=str(source))
    files(root / "data", **{"nan_0.pt": "", "nan.log": "sample 0 has reward NaN", "long.log": "65536 tokens"})
    log(root / "base" / "train.log", UPDATES)
    log(root / "resume" / "train.log", {2: UPDATES[2]})
    log(root / "old_checkpoint" / "train.log", {2: UPDATES[2]}, scale=1.3)
    log(root / "data_parallel" / "train.log", {n: UPDATES[n] for n in (0, 1)}, scale=1.004)
    log(root / "longest_sample" / "train.log", {0: UPDATES[0]})
    log(root / "kill_save" / "train.log", {1: UPDATES[1]})
    for name in ("base", "resume", "old_checkpoint", "data_parallel", "longest_sample", "kill_save"):
        files(root / name, exit="0", end_time="100")
    files(root / "nan", exit="1", end_time="100")
    files(root / "kill_save", tracker_after_kill="0", ckpt_after_kill="iter_0000000\niter_0000001\nlatest_checkpointed_iteration.txt\nrollout", tracker_after_resume="1")
    files(root / "kill_rank", exit="1", inject_time="1000 4242", end_time="1100")
    files(root / "freeze", exit="143", inject_time="1000 4243", end_time="1700",
          **{"watch.log": "no update for 600 s; stopping replay"})
    files(root / "sigterm", exit="143", inject_time="1000", left_after_90s="")
    return root


def test_a_healthy_replay_passes(tmp_path):
    result = replay_check.verdict(healthy(tmp_path))
    assert {n for n, c in result["cases"].items() if not c["passed"]} == set()
    assert result["passed"]


def test_a_resume_that_trains_differently_fails(tmp_path):
    root = healthy(tmp_path)
    log(root / "resume" / "train.log", {2: UPDATES[2]}, scale=1.1)
    case = replay_check.verdict(root)["cases"]["resume"]
    assert not case["passed"] and case["update_2"]["relative_difference"]["train/loss"] > 0.05


def test_a_nan_update_that_was_saved_fails_and_a_finite_one_passes(tmp_path):
    root = healthy(tmp_path)
    log(root / "nan" / "train.log", {0: {"train/loss": float("nan"), "train/grad_norm": float("nan")}})
    files(root / "nan", exit="0", tracker="0")
    assert not replay_check.verdict(root)["cases"]["nan"]["passed"]
    log(root / "nan" / "train.log", {0: UPDATES[0]})
    assert replay_check.verdict(root)["cases"]["nan"]["passed"]


def test_drills_fail_when_the_job_hangs_or_leaves_processes(tmp_path):
    root = healthy(tmp_path)
    files(root / "kill_rank", end_time=str(1000 + 2000))
    files(root / "sigterm", left_after_90s="4321 raylet")
    files(root / "freeze", **{"watch.log": ""})
    cases = replay_check.verdict(root)["cases"]
    assert not cases["kill_rank"]["passed"] and not cases["sigterm"]["passed"] and not cases["freeze"]["passed"]


def test_a_kill_after_the_save_finished_is_inconclusive(tmp_path):
    root = healthy(tmp_path)
    files(root / "kill_save", tracker_after_kill="1")
    case = replay_check.verdict(root)["cases"]["kill_save"]
    assert not case["passed"] and "inconclusive" in case["note"]
