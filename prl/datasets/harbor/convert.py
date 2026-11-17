"""Harbor task directories -> Miles prompt rows for the in-process Harbor agent (``prl.rollout.harbor_agent``).

One row per task dir (``instruction.md`` + ``task.toml``): the prompt is the instruction (Terminus-2 reads it again
from the task dir; Miles only needs a non-empty prompt), ``label`` is empty (the verifier is the reward), and
``metadata.instance_id`` names the task dir under ``HARBOR_TASKS_DIR``.

Usage:
    python -m prl.datasets.harbor.convert --tasks-dir DIR --out OUT.jsonl [--agent terminus-2] [--data-source NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def convert(tasks_dir: str, agent: str, data_source: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(tasks_dir)):
        tdir = os.path.join(tasks_dir, name)
        if not os.path.isfile(os.path.join(tdir, "task.toml")):
            continue
        if not _SAFE_ID.match(name):
            raise ValueError(f"task dir name {name!r} is not a valid instance_id")
        rows.append({
            "prompt": Path(tdir, "instruction.md").read_text().strip(),
            "label": "",
            "metadata": {"instance_id": name, "agent_name": agent, "data_source": data_source},
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--agent", default="terminus-2")
    p.add_argument("--data-source", default="")
    a = p.parse_args()
    rows = convert(a.tasks_dir, a.agent, a.data_source or os.path.basename(os.path.normpath(a.tasks_dir)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
