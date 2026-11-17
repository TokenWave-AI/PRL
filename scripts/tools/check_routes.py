#!/usr/bin/env python3
"""Check the R3 routes of a run's rollout dumps (traces/rollout_data/*.pt, a run with OTA_DUMP_ROUTES=1) against the model:
one row per token but the last, every expert id in range, top-k distinct within a row, and not all zero (SGLang returns
zeros when a runner bypasses the capture). Also counts the rows that repeat their predecessor in every layer, which a
wrong per-turn offset would produce in bulk.
    check_routes.py RUN_DIR --layers 40 --topk 8 --experts 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def check_sample(sample: dict, layers: int, topk: int, experts: int) -> tuple[list[str], dict]:
    routes, tokens = sample.get("rollout_routed_experts"), sample["tokens"]
    if routes is None:
        return ["no routes"], {}
    routes = np.asarray(routes)
    problems = []
    if routes.shape != (len(tokens) - 1, layers, topk):
        problems.append(f"shape {routes.shape}, expected {(len(tokens) - 1, layers, topk)}")
        return problems, {}
    if routes.min() < 0 or routes.max() >= experts:
        problems.append(f"expert ids in [{routes.min()}, {routes.max()}]")
    if not routes.any():
        problems.append("all zeros")
    ordered = np.sort(routes, axis=-1)
    repeats = int((ordered[..., 1:] == ordered[..., :-1]).any(axis=-1).sum())
    if repeats:
        problems.append(f"{repeats} (token, layer) rows select an expert twice")
    same = float((routes[1:] == routes[:-1]).all(axis=(1, 2)).mean()) if len(routes) > 1 else 0.0
    return problems, {"rows": len(routes), "same_as_previous": same, "distinct_experts": len(np.unique(routes))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--experts", type=int, default=256)
    a = p.parse_args()
    dumps = sorted((a.run_dir / "traces" / "rollout_data").glob("*.pt"))
    assert dumps, f"no rollout dumps under {a.run_dir}"
    failed = 0
    for dump in dumps:
        samples = torch.load(dump, weights_only=False)["samples"]
        stats = []
        for i, sample in enumerate(samples):
            problems, s = check_sample(sample, a.layers, a.topk, a.experts)
            stats.append(s)
            if problems:
                failed += 1
                print(f"{dump.name} sample {i}: {'; '.join(problems)}")
        stats = [s for s in stats if s]
        if stats:
            print(f"{dump.name}: {len(samples)} samples, rows mean {np.mean([s['rows'] for s in stats]):.0f} "
                  f"max {max(s['rows'] for s in stats)}, row same as previous {np.mean([s['same_as_previous'] for s in stats]):.3f}, "
                  f"distinct experts per sample mean {np.mean([s['distinct_experts'] for s in stats]):.0f}", flush=True)
    print("routes ok" if not failed else f"{failed} samples FAILED")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
