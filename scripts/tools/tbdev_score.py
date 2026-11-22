"""TB-dev score of one or more Harbor trial dirs (an eval job's harbor_trials, plus any rerun of its capped trials).

Mean verifier reward over every trial. A trial with an exception scores its verifier reward if the verifier ran (e.g.
the agent hit its turn limit and the tests were still graded), else 0; --strict scores every exception 0.
--samples N (the eval's EVAL_N_SAMPLES): with PRL_EVAL_SANDBOX_RETRIES (prl/rollout/harbor_training.py) an
evaluation runs a trial again when it failed on the sandbox service, which leaves the failed attempt's trial dir beside
the new one; a task with more than N trials drops such failed attempts, the earliest first, down to N.
--drop-sandbox-failures: every trial the sandbox service failed is left out and the score is the mean over tasks of each
task's mean, as an evaluation that ran them again would score in expectation; the way to compare an evaluation from
before PRL_EVAL_SANDBOX_RETRIES with a later one.
    python3 scripts/tools/tbdev_score.py [--strict] [--samples 8 | --drop-sandbox-failures] <harbor_trials dir>...
"""

import argparse
import collections
import glob
import json
from pathlib import Path

# as SANDBOX_FAILURES in prl/rollout/harbor_training.py (tests/unit/test_tbdev_score.py checks they agree)
SANDBOX_FAILURES = ("RateLimitException", "SandboxException", "Command ended without an end event",
                    "ReadTimeout", "RemoteProtocolError", "TimeoutException")

ap = argparse.ArgumentParser()
ap.add_argument("--strict", action="store_true")
ap.add_argument("--samples", type=int, default=0)
ap.add_argument("--drop-sandbox-failures", action="store_true")
ap.add_argument("dirs", nargs="+")
args = ap.parse_args()
per, exc = collections.defaultdict(list), collections.Counter()
for d in args.dirs:
    for f in sorted(glob.glob(d + "/**/result.json", recursive=True)):
        j = json.loads(Path(f).read_text())
        if "task_name" not in j and "trial_name" not in j:
            continue  # the job-level result.json
        task = j.get("task_name") or j["trial_name"].rsplit("__", 1)[0]
        rewards = (j.get("verifier_result") or {}).get("rewards") or {}
        e = j.get("exception_info")
        graded = not e or (rewards and not args.strict)
        text = f"{(e or {}).get('exception_type')} {(e or {}).get('exception_message')}"
        failed = bool(e) and not rewards and any(failure in text for failure in SANDBOX_FAILURES)
        per[task].append((j.get("started_at") or "", float(rewards.get("reward", 0) or 0) if graded else 0.0,
                          failed, (e or {}).get("exception_type")))
dropped = 0
if args.samples:
    for task, trials in per.items():
        trials.sort()
        while len(trials) > args.samples and any(t[2] for t in trials):
            trials.remove(next(t for t in trials if t[2]))
            dropped += 1
if args.drop_sandbox_failures:
    for task in list(per):
        dropped += sum(t[2] for t in per[task])
        per[task] = [t for t in per[task] if not t[2]]
        if not per[task]:
            del per[task]  # every trial of it failed on the sandbox service: nothing to score
for trials in per.values():
    exc.update(t[3] for t in trials if t[3])
n, total = sum(map(len, per.values())), sum(t[1] for trials in per.values() for t in trials)
mean = (sum(sum(t[1] for t in trials) / len(trials) for trials in per.values()) / len(per)
        if args.drop_sandbox_failures else total / n)
print(f"tasks={len(per)} trials={n} mean={mean:.4f} trials_per_task={dict(collections.Counter(map(len, per.values())))} "
      f"exceptions={dict(exc)}" + (f" dropped_sandbox_failures={dropped}"
                                   if args.samples or args.drop_sandbox_failures else ""))
