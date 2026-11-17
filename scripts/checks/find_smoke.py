"""The newest accepted framework smoke that vouches for a training job. Called by scripts/lib/chain.sh
(find_smoke): a site submit.sh at submission, the smoke and train_eval steps at run time.

A smoke qualifies when its receipt (acceptance.json) passed for these sources (scripts/checks/grpo_acceptance.py
source_hashes) and this algorithm, and its preflight (preflight.json settings) ran this model at this context and
schedule (ASYNC: the async smoke vouches only for async jobs, a synchronous one only for synchronous jobs): the checks
scripts/chain/grpo_full.sh makes again before training. Prints the smoke's RUN_DIR, or nothing.

Configuration (environment): PRL_ROOT, PRL_EXT, ALGO (grpo | ppo | reinforce_plus_plus), MODEL (configs/models name),
MAX_CONTEXT (the training arm's), ASYNC (default 0).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # grpo_acceptance, when run as a script
from grpo_acceptance import source_hashes


def mismatch(run: Path, receipt: dict, current: dict, algo: str, model: str, context: str, schedule: str = "0") -> str:
    """Why the smoke at ``run`` (its receipt, acceptance.json) does not vouch for a job, or "" when it does. Also the
    check of scripts/chain/grpo_full.sh before training."""
    try:
        settings = json.loads((run / "preflight.json").read_text())["settings"]
    except (OSError, ValueError, KeyError) as exc:
        return f"no preflight settings ({exc})"
    for what, have, want in (
        ("passed", receipt.get("passed") is True, True),
        ("sources", receipt.get("source_hashes") == current, True),
        # receipts from before ADVANTAGE_ESTIMATOR existed are GRPO; a PPO run needs a smoke that trained a critic
        ("algorithm", (receipt.get("algorithm", "grpo"), settings.get("ADVANTAGE_ESTIMATOR", "grpo")), (algo, algo)),
        ("MODEL", settings.get("MODEL"), model),  # its TITO template, parsers, weight sync
        ("MAX_CONTEXT", settings.get("MAX_CONTEXT"), context),
        # the async smoke checked mixed weight versions, a synchronous one a version per batch
        ("ASYNC", settings.get("ASYNC", "0"), schedule),
    ):
        if have != want:
            return f"{what}: {have} != {want}"
    return ""


def find(root: Path, ext: Path, algo: str, model: str, context: str, schedule: str = "0") -> str:
    current, found = source_hashes(root), []
    for receipt in glob.glob(f"{ext}/runs/{algo}_ota_train_smoke_*/acceptance.json"):
        run = Path(receipt).parent
        try:
            r = json.loads(Path(receipt).read_text())
        except (OSError, ValueError):
            continue
        if not mismatch(run, r, current, algo, model, context, schedule):
            found.append((Path(receipt).stat().st_mtime, str(run)))
    return max(found)[1] if found else ""


if __name__ == "__main__":
    env = os.environ
    print(find(Path(env["PRL_ROOT"]), Path(env["PRL_EXT"]), env["ALGO"], env["MODEL"], env["MAX_CONTEXT"],
               env.get("ASYNC", "0")))
