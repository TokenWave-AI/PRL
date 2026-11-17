"""The watchdog of a training run (scripts/lib/guard.sh run_guarded; host Python, no repo imports).

Reads only what this job appends to train.log (from OFFSET: relay jobs share RUN_DIR and append to one train.log, so a
fatal line of an earlier job must not stop a later one) and stops the job (SIGTERM to RUNNER_PID) on:
- the fatal training call-tag signature: a deterministic sample-contract error otherwise refills groups indefinitely;
- no progress: no new update ('train/step') for PRL_STALL_SECS (default 2 x AGENT_TRIAL_TIMEOUT + 30 min: an
  update may wait for a trial (at most AGENT_TRIAL_TIMEOUT) and then for the trial refilling a dropped group). Silent
  hangs held GPU nodes for 4.5 h and 3 h until someone looked. It first sends SIGABRT to this
  node's Ray workers, whose fault handler (PYTHONFAULTHANDLER=1) writes every thread's stack to their logs, and copies
  those to RUN_DIR/stall_stacks_<JOB_ID>.txt;
- mass discards: more groups dropped than kept (rollout/dynamic_filter/drop_group_has_aborted > ROLLOUT_BATCH) in
  PRL_DROP_STEPS (default 3) consecutive updates, e.g. a sandbox outage or rate limit (one run's step 5: 19 of a
  32-group batch).
Every stop is also written to RUN_DIR/watch.log. Usage: train_watch.py TRAIN_LOG OFFSET RUNNER_PID WHAT
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

CALL_TAG = "Missing/unknown training call tag"
STEP = re.compile(r"'train/step': (\d+)")
PERF = re.compile(r"perf (\d+): \{")
DROPPED = re.compile(r"'rollout/dynamic_filter/drop_group_has_aborted': (\d+)")


class Watch:
    def __init__(self, batch: int, stall_secs: float, drop_steps: int, now: float) -> None:
        self.batch, self.stall_secs, self.drop_steps = batch, stall_secs, drop_steps
        self.progress = now
        self.streak = 0

    def line(self, text: str, now: float) -> str | None:
        """A reason to stop, or None."""
        if CALL_TAG in text:
            return "fatal training call-tag contract"
        if STEP.search(text):
            self.progress = now
        if PERF.search(text):
            dropped = int(m[1]) if (m := DROPPED.search(text)) else 0
            self.streak = self.streak + 1 if dropped > self.batch else 0
            if self.streak >= self.drop_steps:
                return (f"{self.streak} updates in a row dropped more groups than they kept (last: {dropped} dropped,"
                        f" batch {self.batch})")
        return None

    def stalled(self, now: float) -> str | None:
        if now - self.progress > self.stall_secs:
            return f"no update for {int(now - self.progress)} s (limit {int(self.stall_secs)} s)"
        return None


def _dump_stacks(run_dir: Path) -> None:
    subprocess.run(["pkill", "-ABRT", "-u", os.environ.get("USER", ""), "-f", "^ray::"], check=False)
    time.sleep(30)
    logs = Path(os.environ.get("PRL_SCRATCH", "/nonexistent")) / "ray_tmp" / "session_latest" / "logs"
    out = run_dir / f"stall_stacks_{os.environ.get('JOB_ID', 'local')}.txt"
    with out.open("w") as f:
        for err in sorted(logs.glob("*.err")):
            text = err.read_text(errors="replace")
            if "Fatal Python error" in text:
                f.write(f"=== {err}\n" + "\n".join(text.splitlines()[-400:]) + "\n")


def main() -> None:
    log, offset, runner, what = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    run_dir = log.parent
    trial = float(os.environ.get("AGENT_TRIAL_TIMEOUT") or 7800)
    watch = Watch(int(os.environ.get("ROLLOUT_BATCH") or 64), float(os.environ.get("PRL_STALL_SECS") or 2 * trial + 1800),
                  int(os.environ.get("PRL_DROP_STEPS") or 3), time.time())
    interval = float(os.environ.get("PRL_WATCH_INTERVAL") or 60)
    position, rest, reason = offset, b"", None
    while reason is None:
        try:
            os.kill(runner, 0)
        except ProcessLookupError:
            return
        time.sleep(interval)
        if log.exists() and log.stat().st_size > position:
            with log.open("rb") as f:
                f.seek(position)
                chunk = f.read()
            position += len(chunk)
            *lines, rest = (rest + chunk).split(b"\n")
            for raw in lines:
                if reason := watch.line(raw.decode(errors="replace"), time.time()):
                    break
        if reason is None and (reason := watch.stalled(time.time())):
            _dump_stacks(run_dir)
    message = f"[prl] {reason}; stopping {what}"
    print(message, file=sys.stderr, flush=True)
    with (run_dir / "watch.log").open("a") as f:
        f.write(f"{time.strftime('%FT%T')} job {os.environ.get('JOB_ID', 'local')}: {message}\n")
    try:
        os.kill(runner, signal.SIGTERM)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    main()
