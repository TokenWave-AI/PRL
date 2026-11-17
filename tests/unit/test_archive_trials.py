"""scripts/lib/archive_trials.sh (for file-count quotas): a finished job's trial directories become one tar each,
removed only after the tar lists every file; off with PRL_ARCHIVE_TRIALS=0. Bash + stdlib only."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def archive(*dirs: Path, **env: str) -> str:
    script = 'source scripts/lib/archive_trials.sh; archive_trials "$@"'
    return subprocess.run(["bash", "-c", script, "_", *map(str, dirs)], cwd=ROOT, capture_output=True, text=True,
                          check=True, env={"PATH": "/usr/bin:/bin", **env}).stdout


def trials(root: Path, n: int) -> Path:
    d = root / "harbor_trials" / "job_123"
    for i in range(n):
        (d / f"task_{i}" / "agent").mkdir(parents=True)
        (d / f"task_{i}" / "result.json").write_text("{}")
        (d / f"task_{i}" / "agent" / "trajectory.json").write_text("[]")
    return d


def test_packs_and_removes_the_directory(tmp_path) -> None:
    d = trials(tmp_path, 3)
    out = archive(d)
    tar = d.with_name("job_123.tar")
    assert not d.exists() and tar.exists() and not tar.with_name("job_123.tar.part").exists()
    with tarfile.open(tar) as t:
        assert sorted(m.name for m in t.getmembers() if m.isfile()) == sorted(
            f"job_123/task_{i}/{f}" for i in range(3) for f in ("result.json", "agent/trajectory.json"))
    assert "6 files -> job_123.tar" in out


def test_switch_off_missing_empty_and_existing_tar(tmp_path) -> None:
    d = trials(tmp_path, 2)
    assert "off" in archive(d, PRL_ARCHIVE_TRIALS="0") and d.exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    archive(empty, tmp_path / "missing")  # nothing to pack: no tar, no error
    assert empty.exists() and not (tmp_path / "empty.tar").exists()
    d.with_name("job_123.tar").write_text("an older tar")
    assert "exists" in archive(d) and d.exists()  # never overwrites a tar
