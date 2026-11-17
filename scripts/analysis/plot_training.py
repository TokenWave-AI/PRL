"""Training curves of OTA runs: rollout reward per update, and for runs with a trajectory transform (e.g. ATLAS) the
share of trained tokens it kept (and, for ATLAS, of steps marked critical).

Reward: ``rollout/raw_reward`` from each run's train.log (the last record per rollout, as a relay may repeat one).
Transform: ``prl_transform`` in the trained samples' metadata (traces/rollout_data/<rollout>.pt; older ATLAS
runs: ``prl_atlas``), counted once per trial. Trials the judge did not label are left out of the
critical-step share.
Needs torch and matplotlib (the Miles image):
    python3 scripts/analysis/plot_training.py --out figures/ota_9b_training.png \
        "sync GRPO=$PRL_EXT/runs/<run>" ...
"""

import argparse
import ast
import glob
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RECORD = re.compile(r"\brollout (\d+): (\{.*\})")


def rewards(run_dir):
    per = {}
    for line in Path(run_dir, "train.log").read_text(errors="replace").splitlines(keepends=True):
        m = RECORD.search(line)
        if m and "'rollout/raw_reward'" in m.group(2):
            per[int(m.group(1))] = ast.literal_eval(m.group(2))["rollout/raw_reward"]
    return sorted(per.items())


def _trial(md):
    """(steps, critical steps or None, trained tokens before, after) of one trial's record, or None."""
    if "prl_transform" in md:
        r = md["prl_transform"]
        a = next((e.get("record", {}).get("atlas") for e in r["chain"] if e["transform"] == "atlas"), None) or {}
        critical = a.get("critical")
        return a.get("steps"), None if critical is None else len(critical), r["trained_tokens_before"], \
            r["trained_tokens_after"]
    if "prl_atlas" in md:
        a = md["prl_atlas"]
        a = ast.literal_eval(a) if isinstance(a, str) else a
        return a["steps"], a["critical"], a["trained_tokens"], a["kept_tokens"]
    return None


def transform_stats(run_dir):
    import torch

    rows = []
    for f in glob.glob(os.path.join(run_dir, "traces/rollout_data/*.pt")):
        trials = {}
        for s in torch.load(f, weights_only=False)["samples"]:
            md = s.get("metadata") or {}
            t = _trial(md)
            if t is not None:
                trials[md.get("trial_dir")] = t
        if trials:
            v = trials.values()
            labeled = [t for t in v if t[0] and t[1] is not None]
            crit = sum(t[1] for t in labeled) / sum(t[0] for t in labeled) if labeled else float("nan")
            rows.append((int(os.path.basename(f)[:-3]), crit, sum(t[3] for t in v) / sum(t[2] for t in v), len(trials)))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs]
    masks = [(label, transform_stats(d)) for label, d in runs]
    masks = [(label, rows) for label, rows in masks if rows]
    fig, axes = plt.subplots(1, 1 + bool(masks), figsize=(6.5 * (1 + bool(masks)), 4.2), squeeze=False)
    ax = axes[0][0]
    for label, d in runs:
        steps, values = zip(*rewards(d))
        ax.plot(steps, values, marker=".", label=label)
        print(f"{label}: reward first {values[0]:.3f}, mean of the last 8 {sum(values[-8:]) / 8:.3f}")
    ax.set(xlabel="update", ylabel="mean rollout reward", title="Training reward (64 tasks x 8 per update)")
    ax.grid(alpha=0.3)
    ax.legend()
    if masks:
        ax = axes[0][1]
        for label, rows in masks:
            steps, crit, kept, n = zip(*rows)
            if not all(math.isnan(c) for c in crit):  # critical steps: ATLAS only
                ax.plot(steps, crit, marker=".", label=f"{label}: steps critical")
            ax.plot(steps, kept, marker=".", label=f"{label}: trained tokens kept")
            print(f"{label}: {sum(n)} trials; steps critical {min(crit):.3f}-{max(crit):.3f}, "
                  f"tokens kept {min(kept):.3f}-{max(kept):.3f}")  # nan: not ATLAS
        ax.set(xlabel="update", ylabel="fraction", ylim=(0, 1), title="Trajectory transform: share kept in the loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
