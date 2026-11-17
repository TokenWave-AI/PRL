"""The judge of the chain step replay (scripts/chain/replay_check.sh): train-only runs on an accepted smoke's batches.

    replay_check.py nan SRC.pt OUT.pt   (Miles image: torch) the batch with one sample's reward NaN
    replay_check.py verdict R           (host Python) judge the cases under R (= $PRL_EXT/runs/replay_<job>) into
                                        R/replay.json; exit 1 if one failed

Each case directory holds its train.log (Miles' "step N: {...}" lines: the metrics of update N) and what the step
recorded: exit (the run's status), end_time / inject_time (epoch seconds; inject_time also names the signalled pid),
tracker (the checkpoint tracker when the run ended), and per drill tracker_after_kill, ckpt_after_kill,
tracker_after_resume, left_after_90s. The updates are compared metric by metric: |a - b| <= atol + rtol * max(|a|, |b|).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

METRICS = ("train/loss", "train/pg_loss", "train/entropy_loss", "train/train_rollout_logprob_abs_diff",
           "train/train_rollout_kl", "train/kl_loss", "train/grad_norm")
# The same weights and batch give the same forward (update 0) up to kernel nondeterminism; a later update also carries
# the earlier updates' (bf16, reduction-order) differences, and 4 data-parallel ranks reduce in another order
SAME = (1e-2, 1e-4)
LATER = (5e-2, 5e-4)
STEP = re.compile(r"step (\d+): (\{'train/.*\})")
PAIR = re.compile(r"'([\w/\-]+)': ([^,}]+)")


def updates(log: Path) -> dict[int, dict[str, float]]:
    """The metrics of every update in a train.log (the last line of each; Miles logs it twice)."""
    found: dict[int, dict[str, float]] = {}
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if m := STEP.search(line):
                found[int(m[1])] = {k: float(v) for k, v in PAIR.findall(m[2]) if k in METRICS}
    return found


def finite(metrics: dict[str, float] | None) -> bool:
    return bool(metrics) and all(math.isfinite(v) for v in metrics.values())


def compare(a: dict[str, float] | None, b: dict[str, float] | None, tol: tuple[float, float]) -> tuple[bool, dict]:
    """Whether two updates agree on every metric both logged, and each metric's relative difference."""
    if not a or not b:
        return False, {"missing": "first" if not a else "second"}
    rtol, atol = tol
    diffs, ok = {}, True
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        diffs[key] = abs(x - y) / max(abs(x), abs(y), 1e-12)
        ok &= math.isfinite(x) and math.isfinite(y) and abs(x - y) <= atol + rtol * max(abs(x), abs(y))
    return ok, {"relative_difference": diffs, "first": a, "second": b}


def read(case: Path, name: str) -> str | None:
    return (case / name).read_text().strip() if (case / name).exists() else None


def ended_after(case: Path) -> float | None:
    """Seconds from the injection to the end of the run."""
    inject, end = read(case, "inject_time"), read(case, "end_time")
    return float(end) - float(inject.split()[0]) if inject and end else None


def verdict(root: Path) -> dict:
    source = Path(read(root, "source") or "")
    smoke = updates(source / "train.log")
    run = {name: updates(root / name / "train.log") for name in
           ("base", "resume", "old_checkpoint", "data_parallel", "longest_sample", "nan", "kill_save", "kill_rank",
            "freeze", "sigterm")}
    code = {name: read(root / name, "exit") for name in run}
    cases: dict[str, dict] = {}

    ok0, d0 = compare(run["base"].get(0), smoke.get(0), SAME)
    ok1, d1 = compare(run["base"].get(1), smoke.get(1), LATER)
    cases["base"] = {"what": "the smoke's first two updates replayed (H2: the reference curve)",
                     "passed": code["base"] == "0" and ok0 and ok1 and finite(run["base"].get(2)),
                     "update_0": d0, "update_1": d1, "updates": run["base"]}
    ok, d = compare(run["resume"].get(2), run["base"].get(2), SAME)
    cases["resume"] = {"what": "checkpoint 1 resumed: update 2 as without the restart (H1)",
                       "passed": code["resume"] == "0" and ok, "update_2": d}
    cases["old_checkpoint"] = {"what": "the smoke's checkpoint (older code) resumed for one update (H4)",
                               "passed": code["old_checkpoint"] == "0" and finite(run["old_checkpoint"].get(2)),
                               "update_2": run["old_checkpoint"].get(2)}
    ok0, d0 = compare(run["data_parallel"].get(0), run["base"].get(0), LATER)
    ok1, d1 = compare(run["data_parallel"].get(1), run["base"].get(1), LATER)
    cases["data_parallel"] = {"what": "4 data-parallel trainer ranks instead of 2 (C2)",
                              "passed": code["data_parallel"] == "0" and ok0 and ok1, "update_0": d0, "update_1": d1}
    cases["longest_sample"] = {"what": "one update with a sample of MAX_CONTEXT tokens (A4)",
                               "passed": code["longest_sample"] == "0" and finite(run["longest_sample"].get(0)),
                               "update_0": run["longest_sample"].get(0), "batch": read(root / "data", "long.log")}
    nan = run["nan"].get(0)
    made = (root / "data" / "nan_0.pt").exists()
    # stopped before saving, or trained a finite update (the NaN sample left out); never a NaN update saved
    cases["nan"] = {"what": "a batch with a NaN reward (C6/I4)", "exit": code["nan"], "update_0": nan,
                    "tracker": read(root / "nan", "tracker"), "batch": read(root / "data", "nan.log"),
                    "passed": made and (finite(nan) if code["nan"] == "0" else read(root / "nan", "tracker") is None)}

    killed, resumed = read(root / "kill_save", "tracker_after_kill"), read(root / "kill_save", "tracker_after_resume")
    mid_save = "iter_0000001" in (read(root / "kill_save", "ckpt_after_kill") or "") and killed == "0"
    ok, d = compare(run["kill_save"].get(1), run["base"].get(1), LATER)
    cases["kill_save"] = {"what": "killed while saving checkpoint 1, then resumed (I1)",
                          "killed_mid_save": mid_save, "tracker_after_kill": killed, "tracker_after_resume": resumed,
                          "update_1_after_resume": d, "exit_after_resume": code["kill_save"],
                          "passed": mid_save and code["kill_save"] == "0" and resumed == "1" and ok}
    if killed == "1":
        cases["kill_save"]["note"] = "inconclusive: the save had finished before the kill"

    for name, limit, what in (("kill_rank", 1200 + 300, "a trainer rank SIGKILLed: the job ends (I2)"),
                              ("freeze", 900 + 600, "a trainer rank SIGSTOPped: the watchdog stops the job (I3)")):
        case = root / name
        after, inject = ended_after(case), read(case, "inject_time") or ""
        watch = read(case, "watch.log") or ""
        cases[name] = {"what": what, "seconds_to_end": after, "signalled": inject, "exit": code[name],
                       "watchdog": watch, "stall_stacks": sorted(p.name for p in case.glob("stall_stacks_*"))}
        cases[name]["passed"] = len(inject.split()) == 2 and after is not None and after <= limit and code[name] != "0"
        if name == "freeze":
            cases[name]["passed"] &= "no update" in watch
    left = read(root / "sigterm", "left_after_90s")
    cases["sigterm"] = {"what": "SIGTERM to the job's process group: nothing left 90 s later (I5)",
                        "left_after_90s": left, "exit": code["sigterm"],
                        "passed": read(root / "sigterm", "inject_time") is not None and not left
                        and code["sigterm"] != "0"}
    return {"source": str(source), "passed": all(c["passed"] for c in cases.values()), "cases": cases}


def make_nan(src: str, out: str) -> None:
    import torch
    dump = torch.load(src, weights_only=False)
    sample = dump["samples"][0]
    sample["reward"] = float("nan")
    torch.save(dump, out)
    print(f"{src} -> {out}: sample 0 (group {sample.get('group_index')}) has reward NaN")


def main() -> None:
    if sys.argv[1] == "nan":
        make_nan(sys.argv[2], sys.argv[3])
        return
    root = Path(sys.argv[2])
    result = verdict(root)
    (root / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, case in result["cases"].items():
        print(f"[replay] {name}: {'passed' if case['passed'] else 'FAILED'} ({case['what']})", flush=True)
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
