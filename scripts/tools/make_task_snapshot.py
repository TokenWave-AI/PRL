"""Make a task snapshot: the task directories an arm runs on, beside the manifest of their sandbox templates.

    python3 scripts/tools/make_task_snapshot.py --tasks-dir $PRL_EXT/data/harbor_tasks \
        --rows $PRL_EXT/data/otagent_rl/train.jsonl --out $PRL_EXT/task_snapshots/otagent728 --tmux --smoke 8
    python3 scripts/tools/make_task_snapshot.py --tasks-dir $PRL_EXT/data/harbor_tasks \
        --rows $PRL_EXT/data/ot_tbdev/tbdev.jsonl --out $PRL_EXT/task_snapshots/tbdev70 --tmux

A snapshot is a directory the arms read through PRL_TRAIN_SNAPSHOT / PRL_EVAL_SNAPSHOT (configs/arms/common_ota.env):
    tasks/          real copies of the task directories the rows name (metadata.instance_id), permissions kept
    smoke.jsonl     with --smoke N: the first N rows, the prompts of the framework smoke (configs/arms/ota_train_smoke.env)
    manifest.json   task -> ready sandbox template: build the images from tasks/ and register them (docs/sandboxes.md)
    contract/       oracle / no-op results of CHAIN=task_check (scripts/chain/task_smoke.sh); the smoke's preflight
                    requires those of its tasks
A snapshot is never changed after its templates are built: a template is bound to the context hash of its task's
environment/ (prl/sandbox/template_manifest.py), so a new task set or a new image gets a new snapshot directory.

--tmux: Terminus-2 drives the shell through tmux. A task image without it installs it at the start of every trial (about
10 s, and a failed or timed-out install loses the trial: prl/rollout/harbor_fixes.py), so --tmux appends an apt-get
install of tmux to every environment/Dockerfile that does not mention tmux, as root, then restores the Dockerfile's
last USER. The task images are Debian / Ubuntu based; the image then differs from the task's own by that one layer.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

TMUX = """
# make_task_snapshot.py --tmux: Terminus-2 drives the shell through tmux; installed here instead of in every trial
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux
"""


def add_tmux(dockerfile: Path) -> bool:
    """Append the tmux layer unless the Dockerfile mentions tmux already; True when it was added."""
    text = dockerfile.read_text()
    if "tmux" in text:
        return False
    users = re.findall(r"^\s*USER\s+(\S+)", text, re.MULTILINE | re.IGNORECASE)
    restore = f"USER {users[-1]}\n" if users and users[-1] not in ("root", "0") else ""
    dockerfile.write_text(text.rstrip("\n") + "\n" + TMUX + restore)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tasks-dir", type=Path, required=True, help="Harbor task directories (prepare_otagent.sh)")
    parser.add_argument("--rows", type=Path, required=True, help="Miles prompt rows (JSONL) selecting the tasks")
    parser.add_argument("--out", type=Path, required=True, help="a new snapshot directory")
    parser.add_argument("--tmux", action="store_true", help="add a tmux layer to images without tmux")
    parser.add_argument("--smoke", type=int, default=0, help="write the first N rows as smoke.jsonl")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists: a snapshot is never changed, choose a new directory")
    lines = [line for line in args.rows.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    names = [row["metadata"]["instance_id"] for row in rows]
    if len(set(names)) != len(names):
        raise SystemExit(f"{args.rows}: a task appears twice")
    missing = [name for name in names if not (args.tasks_dir / name / "environment").is_dir()]
    if missing:
        raise SystemExit(f"{len(missing)} tasks of {args.rows} have no environment/ in {args.tasks_dir}: {missing[:5]}")

    tasks = args.out / "tasks"
    tasks.mkdir(parents=True)
    added = 0
    for name in names:
        # real directories (Miles rejects task paths that resolve elsewhere); links inside a task stay links
        shutil.copytree(args.tasks_dir / name, tasks / name, symlinks=True)
        if args.tmux:
            added += add_tmux(tasks / name / "environment" / "Dockerfile")
    if args.smoke:
        (args.out / "smoke.jsonl").write_text("".join(line + "\n" for line in lines[:args.smoke]))
    print(f"[snapshot] {args.out}: {len(names)} tasks, tmux added to {added}, {min(args.smoke, len(rows))} smoke rows")


if __name__ == "__main__":
    main()
