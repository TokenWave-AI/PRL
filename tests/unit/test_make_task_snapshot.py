"""scripts/tools/make_task_snapshot.py: real copies of the tasks the rows name, the tmux layer only where an image lacks
tmux (restoring the Dockerfile's last USER), the smoke rows verbatim, and a snapshot directory that is never reused."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "tools" / "make_task_snapshot.py"


def _task(root: Path, name: str, dockerfile: str) -> None:
    (root / name / "environment").mkdir(parents=True)
    (root / name / "environment" / "Dockerfile").write_text(dockerfile)
    (root / name / "tests").mkdir()
    (root / name / "tests" / "test.sh").write_text("exit 0\n")
    (root / name / "tests" / "test.sh").chmod(0o755)


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), "--tasks-dir", str(tmp_path / "all"), "--rows",
                           str(tmp_path / "rows.jsonl"), *args], capture_output=True, text=True, check=False)


def test_snapshot(tmp_path: Path) -> None:
    all_tasks = tmp_path / "all"
    _task(all_tasks, "plain", "FROM ubuntu:24.04\nWORKDIR /app\n")
    _task(all_tasks, "has_tmux", "FROM ubuntu:24.04\nRUN apt-get install -y tmux\n")
    _task(all_tasks, "as_user", "FROM ubuntu:24.04\nUSER agent\nWORKDIR /home/agent\n")
    _task(all_tasks, "unused", "FROM ubuntu:24.04\n")
    lines = [json.dumps({"prompt": f"p{n}", "metadata": {"instance_id": name}})
             for n, name in enumerate(["plain", "has_tmux", "as_user"])]
    (tmp_path / "rows.jsonl").write_text("\n".join(lines) + "\n")
    out = tmp_path / "snap"

    done = _run(tmp_path, "--out", str(out), "--tmux", "--smoke", "2")
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in (out / "tasks").iterdir()) == ["as_user", "has_tmux", "plain"]
    assert (out / "smoke.jsonl").read_text() == "\n".join(lines[:2]) + "\n"
    plain = (out / "tasks" / "plain" / "environment" / "Dockerfile").read_text()
    assert plain.startswith("FROM ubuntu:24.04\nWORKDIR /app\n") and "apt-get install" in plain and "tmux" in plain
    assert plain.rstrip().endswith("tmux")  # root, as the Dockerfile left it
    assert (out / "tasks" / "has_tmux" / "environment" / "Dockerfile").read_text() == (
        all_tasks / "has_tmux" / "environment" / "Dockerfile").read_text()
    as_user = (out / "tasks" / "as_user" / "environment" / "Dockerfile").read_text()
    assert "USER root\nRUN apt-get" in as_user and as_user.endswith("USER agent\n")
    assert (out / "tasks" / "plain" / "tests" / "test.sh").stat().st_mode & 0o777 == 0o755

    again = _run(tmp_path, "--out", str(out))
    assert again.returncode != 0 and "never changed" in again.stderr


def test_a_task_without_its_directory_stops(tmp_path: Path) -> None:
    (tmp_path / "all").mkdir()
    (tmp_path / "rows.jsonl").write_text(json.dumps({"prompt": "p", "metadata": {"instance_id": "gone"}}) + "\n")
    done = _run(tmp_path, "--out", str(tmp_path / "snap"))
    assert done.returncode != 0 and "gone" in done.stderr and not (tmp_path / "snap").exists()


def test_output_dir_only_where_the_tests_use_it(tmp_path: Path) -> None:
    all_tasks = tmp_path / "all"
    _task(all_tasks, "writes", "FROM ubuntu:24.04\nWORKDIR /workspace\n")
    (all_tasks / "writes" / "tests" / "test.sh").write_text('OUTPUT="/output/command_capture.txt"\n')
    _task(all_tasks, "as_user", "FROM ubuntu:24.04\nUSER agent\n")
    (all_tasks / "as_user" / "tests" / "test.sh").write_text("cat /output/x\n")
    _task(all_tasks, "has_it", "FROM ubuntu:24.04\nRUN mkdir -p /output\n")
    (all_tasks / "has_it" / "tests" / "test.sh").write_text("cat /output/x\n")
    _task(all_tasks, "no_output", "FROM ubuntu:24.04\n")
    names = ["writes", "as_user", "has_it", "no_output"]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps({"prompt": n, "metadata": {"instance_id": n}}) + "\n"
                                                 for n in names))
    done = _run(tmp_path, "--out", str(tmp_path / "snap"), "--output-dir", "--tmux")
    assert done.returncode == 0, done.stderr
    assert "/output added to 2, tmux added to 4" in done.stdout
    dockerfile = {n: (tmp_path / "snap" / "tasks" / n / "environment" / "Dockerfile").read_text() for n in names}
    assert dockerfile["writes"].index("RUN mkdir -p /output") < dockerfile["writes"].index("tmux")  # mkdir first
    assert "USER root\nRUN mkdir -p /output\nUSER agent\n" in dockerfile["as_user"]
    assert dockerfile["as_user"].endswith("USER agent\n")
    assert dockerfile["has_it"].count("/output") == 1 and "mkdir -p /output" not in dockerfile["no_output"]
