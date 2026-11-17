"""scripts/checks/train_watch.py: the training watchdog reads only this job's lines of train.log."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/checks/train_watch.py"
spec = importlib.util.spec_from_file_location("train_watch", SCRIPT)
train_watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_watch)

STEP = "step 3: {'train/loss': 0.1, 'train/step': 3}"


def perf(dropped: int) -> str:
    return f"metrics.py:93 - perf 3: {{'rollout/dynamic_filter/drop_group_has_aborted': {dropped}, 'rollout/x': 1}}"


def test_an_update_is_progress_and_a_long_silence_is_a_stall():
    watch = train_watch.Watch(batch=64, stall_secs=100, drop_steps=3, now=0)
    assert watch.stalled(99) is None
    assert watch.line(STEP, 90) is None
    assert watch.stalled(189) is None
    assert "no update" in watch.stalled(191)


def test_three_updates_dropping_more_than_they_keep_stop_the_run():
    watch = train_watch.Watch(batch=64, stall_secs=1e9, drop_steps=3, now=0)
    assert watch.line(perf(65), 0) is None
    assert watch.line(perf(65), 0) is None
    assert watch.line(perf(64), 0) is None  # as many dropped as kept: the streak restarts
    assert watch.line("perf 3: {'rollout/raw_reward_std': 0.4}", 0) is None  # no drops at all
    assert watch.line(perf(100), 0) is None and watch.line(perf(100), 0) is None
    assert "3 updates in a row" in watch.line(perf(100), 0)


def test_the_call_tag_signature_stops_the_run():
    watch = train_watch.Watch(batch=64, stall_secs=1e9, drop_steps=3, now=0)
    assert "call-tag" in watch.line("ValueError: Missing/unknown training call tag 'x'", 0)


def _watch(log: Path, offset: int, runner: subprocess.Popen) -> subprocess.CompletedProcess:
    env = {**os.environ, "PRL_WATCH_INTERVAL": "0.2", "ROLLOUT_BATCH": "8", "PRL_STALL_SECS": "1000"}
    return subprocess.run([sys.executable, str(SCRIPT), str(log), str(offset), str(runner.pid), "smoke"], env=env,
                          check=False, capture_output=True, text=True, timeout=30)


def test_a_fatal_line_of_an_earlier_relay_job_does_not_stop_this_one(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("earlier job: Missing/unknown training call tag 'x'\n")
    runner = subprocess.Popen(["sleep", "2"])
    threading.Thread(target=runner.wait).start()  # reaped as bash's `wait` does: a zombie would still answer kill -0
    started = time.time()
    result = _watch(log, log.stat().st_size, runner)
    assert result.returncode == 0 and "stopping" not in result.stderr
    assert time.time() - started >= 1.5  # it watched until the job ended by itself
    assert not (tmp_path / "watch.log").exists()


def test_a_fatal_line_of_this_job_stops_it(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("earlier job\n")
    offset = log.stat().st_size
    runner = subprocess.Popen(["sleep", "30"])
    with log.open("a") as f:
        f.write("ValueError: Missing/unknown training call tag 'x'\n")
    result = _watch(log, offset, runner)
    assert "fatal training call-tag contract; stopping smoke" in result.stderr
    assert runner.wait(timeout=5) == -15  # SIGTERM
    assert "call-tag" in (tmp_path / "watch.log").read_text()
