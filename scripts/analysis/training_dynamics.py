"""Training dynamics of OTA runs: trainer/rollout metrics per update and the per-trial length distribution by stage.

Per update, from train.log: the trainer's ``step N`` record (KL, grad norm, clip fractions, train/rollout
logprob gap) and the rollout's ``perf N`` record (zero-std groups, staleness). From the trained samples
(traces/rollout_data/<rollout>.pt), per trial: output tokens, turns, reward, and the sampled tokens' mean negative
log-probability under the rollout policy over every generated token (temperature 1: an estimate of the policy's
entropy that does not depend on a token mask; the trainer logs ``train/entropy_loss`` as 0 with no entropy term, and
``train/pg_clipfrac`` is 0 with one optimizer step per rollout). Generated tokens = the trial's trained tokens before masking (``prl_atlas``
trained_tokens for ATLAS, ``prl_transform`` trained_tokens_before for a transform, else the loss mask); non-generated tokens carry log-prob 0.
Needs torch and matplotlib (the Miles image); the per-trial table is cached (--cache):
    python3 scripts/analysis/training_dynamics.py --out figures/dyn "async GRPO=$PRL_EXT/runs/<run>" ...
"""

import argparse
import ast
import collections
import glob
import json
import os
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

STEP_KEYS = ["train/entropy_loss", "train/kl_loss", "train/grad_norm", "train/pg_clipfrac", "train/ppo_kl",
             "train/train_rollout_logprob_abs_diff", "train/tis_clipfrac"]
PERF_KEYS = ["rollout/zero_std/all_zero_percentage", "rollout/zero_std/all_one_percentage",
             "rollout/fully_async/avg_staleness", "rollout/raw_reward_unfiltered"]


def log_metrics(run_dir):
    per = collections.defaultdict(dict)
    for line in Path(run_dir, "train.log").read_text(errors="replace").splitlines(keepends=True):
        m = re.search(r"\b(step|perf) (\d+): \{", line)
        if not m:
            continue
        for k in STEP_KEYS + PERF_KEYS:
            v = re.search(rf"'{re.escape(k)}': ([0-9.e+-]+|nan)", line)
            if v:
                per[int(m.group(2))][k] = float(v.group(1))
    return per


def trials(run_dir):
    import torch

    rows = []
    for f in glob.glob(os.path.join(run_dir, "traces/rollout_data/*.pt")):
        rid = int(os.path.basename(f)[:-3])
        by = collections.defaultdict(list)
        for s in torch.load(f, weights_only=False)["samples"]:
            by[(s.get("metadata") or {}).get("trial_dir")].append(s)
        for ss in by.values():
            md = ss[0]["metadata"]
            am = md.get("agent_metrics") or {}
            am = ast.literal_eval(am) if isinstance(am, str) else am
            at = md.get("prl_atlas")
            at = ast.literal_eval(at) if isinstance(at, str) else at
            # a trajectory transform's record (runs with the transform framework; older ATLAS runs: prl_atlas)
            tr = md.get("prl_transform")
            tr = ast.literal_eval(tr) if isinstance(tr, str) else tr
            nll = sum(-x for s in ss for x in s["rollout_log_probs"])
            kept = sum(sum(s["loss_mask"]) for s in ss)
            kept_nll = sum(-x for s in ss for x, m in zip(s["rollout_log_probs"], s["loss_mask"]) if m)
            gen = at["trained_tokens"] if at else tr["trained_tokens_before"] if tr else kept
            if tr and (tr.get("tokens_changed") or tr.get("dropped_samples")):
                # rewritten tokens were not sampled; a dropped sample took its tokens: only the trained ones count
                gen, nll = kept, kept_nll
            rows.append({"rollout": rid, "reward": float(ss[0].get("reward") or 0), "out": am.get("n_output_tokens"),
                         "turns": am.get("n_episodes"), "gen": gen, "nll": nll, "kept": kept, "kept_nll": kept_nll,
                         "leaves": len(ss)})
    return rows


def slug(title):
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="per-trial table (default <out>_cache.json)")
    ap.add_argument("--stages", default="0-5,14-19,28-33", help="update ranges compared as training stages")
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs]
    cache_f = args.cache or args.out + "_cache.json"
    cache = json.loads(Path(cache_f).read_text()) if os.path.exists(cache_f) else {}
    for label, d in runs:
        if label not in cache:
            cache[label] = trials(d)
            Path(cache_f).write_text(json.dumps(cache))
    logs = {label: log_metrics(d) for label, d in runs}
    stages = [tuple(map(int, s.split("-"))) for s in args.stages.split(",")]

    # --- per-update metrics
    per_up = {}
    for label, _ in runs:
        g = collections.defaultdict(list)
        for r in cache[label]:
            g[r["rollout"]].append(r)
        per_up[label] = {k: g[k] for k in sorted(g)}
    panels = [("reward", lambda rs, m: st.mean(r["reward"] for r in rs)),
              ("entropy, all generated tokens (rollout NLL)", lambda rs, m: sum(r["nll"] for r in rs) / sum(r["gen"] for r in rs)),
              ("entropy of trained tokens (GRPO: all; ATLAS: kept, dropped dashed)", lambda rs, m: sum(r["kept_nll"] for r in rs) / sum(r["kept"] for r in rs)),
              ("train/kl_loss", lambda rs, m: m.get("train/kl_loss")),
              ("train/grad_norm", lambda rs, m: m.get("train/grad_norm")),
              ("output tokens per trial: p25 / p75 (dashed)", lambda rs, m: q([r["out"] for r in rs if r["out"]], .25)),
              ("train/train_rollout_logprob_abs_diff", lambda rs, m: m.get("train/train_rollout_logprob_abs_diff")),
              ("groups all 0 / all 1 (no signal)", lambda rs, m: (m.get("rollout/zero_std/all_zero_percentage") or 0) + (m.get("rollout/zero_std/all_one_percentage") or 0)),
              ("output tokens per trial: median", lambda rs, m: st.median(r["out"] for r in rs if r["out"])),
              ("output tokens per trial: p90", lambda rs, m: q([r["out"] for r in rs if r["out"]], .9)),
              ("turns per trial: median (p90 dashed)", lambda rs, m: st.median(r["turns"] for r in rs if r["turns"])),
              ("output tokens per turn: median", lambda rs, m: st.median(r["out"] / r["turns"] for r in rs if r["out"] and r["turns"]))]
    def draw(a, title, fn):
        for i, (label, _) in enumerate(runs):
            xs, ys = [], []
            for k, rs in per_up[label].items():
                y = fn(rs, logs[label].get(k, {}))
                if y is not None:
                    xs.append(k)
                    ys.append(y)
            a.plot(xs, ys, ".-", color=f"C{i}", label=label)
            dashed = {"turns": lambda rs: q([r["turns"] for r in rs if r["turns"]], .9),
                      "output tokens per trial: p25": lambda rs: q([r["out"] for r in rs if r["out"]], .75),
                      "entropy of trained": lambda rs: sum(r["nll"] - r["kept_nll"] for r in rs) / sum(r["gen"] - r["kept"] for r in rs)
                      if any(r["kept"] < r["gen"] for r in rs) else None}
            for prefix, g in dashed.items():
                if title.startswith(prefix) and xs and g(per_up[label][xs[0]]) is not None:
                    a.plot(xs, [g(per_up[label][k]) for k in xs], "--", color=f"C{i}", lw=1)
        a.set(title=title, xlabel="update")
        a.grid(alpha=.3)
        a.legend(fontsize=8)

    fig, ax = plt.subplots(3, 4, figsize=(20, 11))
    for (title, fn), a in zip(panels, ax.flat):
        draw(a, title, fn)
    fig.tight_layout()
    fig.savefig(args.out + "_updates.png", dpi=100)
    # each panel on its own as well: <out>_updates/<title>.png
    os.makedirs(args.out + "_updates", exist_ok=True)
    for title, fn in panels:
        f, a = plt.subplots(figsize=(6.4, 4.2))
        draw(a, title, fn)
        f.tight_layout()
        f.savefig(os.path.join(args.out + "_updates", slug(title) + ".png"), dpi=110)
        plt.close(f)

    # --- per-update table
    print("per update (first 4 -> last 8 updates):")
    for title, fn in panels:
        cells = []
        for label, _ in runs:
            ks = list(per_up[label])
            v = lambda ks, fn=fn, label=label: [y for y in (fn(per_up[label][k], logs[label].get(k, {})) for k in ks)
                                                if y is not None]
            a, b = v(ks[:4]), v(ks[-8:])
            if a and b:
                cells.append(f"{label} {st.mean(a):.4g} -> {st.mean(b):.4g}")
        print(f"  {title}: " + "; ".join(cells))
    for label, _ in runs:
        R = cache[label]
        if any(r["kept"] < r["gen"] for r in R):
            for lo, hi in stages:
                rs = [r for r in R if lo <= r["rollout"] <= hi]
                drop = sum(r["gen"] - r["kept"] for r in rs)
                print(f"  {label} updates {lo}-{hi}: NLL kept {sum(r['kept_nll'] for r in rs) / sum(r['kept'] for r in rs):.4f}, "
                      f"dropped {(sum(r['nll'] - r['kept_nll'] for r in rs)) / drop:.4f}")

    # --- length distribution by stage
    print("output tokens per trial by stage: p10 / p25 / median / p75 / p90 / p99 / max | turns median / p90 | tokens per turn median | mean")
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    styles = ["-", "--", ":"]
    for i, (label, _) in enumerate(runs):
        for j, (lo, hi) in enumerate(stages):
            rs = [r for r in cache[label] if lo <= r["rollout"] <= hi and r["out"] and r["turns"]]
            o, t, pt = [r["out"] for r in rs], [r["turns"] for r in rs], [r["out"] / r["turns"] for r in rs]
            print(f"  {label:12} {lo:2}-{hi:2} ({len(rs)}): " + " / ".join(f"{q(o, p):.0f}" for p in (.1, .25, .5, .75, .9, .99))
                  + f" / {max(o)} | {st.median(t):.0f} / {q(t, .9)} | {st.median(pt):.0f} | {st.mean(o):.0f}")
            for a, v in zip(ax, (o, t, pt)):
                v = sorted(v)
                a.plot(v, [(k + 1) / len(v) for k in range(len(v))], styles[j % 3], color=f"C{i}", label=f"{label} {lo}-{hi}")
    for a, t in zip(ax, ("output tokens per trial", "turns per trial", "output tokens per turn")):
        a.set(xscale="log", title=f"training rollouts: {t} (CDF, by stage)", xlabel=t + " (log)")
        a.xaxis.set_minor_formatter(NullFormatter())
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out + "_stages.png", dpi=100)
    os.makedirs(args.out + "_stages", exist_ok=True)
    for a in ax:
        f, b = plt.subplots(figsize=(7, 4.6))
        for line in a.get_lines():
            b.plot(*line.get_data(), line.get_linestyle(), color=line.get_color(), label=line.get_label())
        b.set(xscale="log", title=a.get_title(), xlabel=a.get_xlabel())
        b.xaxis.set_minor_formatter(NullFormatter())
        b.grid(alpha=.3)
        b.legend(fontsize=7)
        f.tight_layout()
        f.savefig(os.path.join(args.out + "_stages", slug(a.get_title()) + ".png"), dpi=110)
        plt.close(f)
    print("wrote", args.out + "_{updates,stages}.png and the single panels in", args.out + "_{updates,stages}/")


if __name__ == "__main__":
    main()
