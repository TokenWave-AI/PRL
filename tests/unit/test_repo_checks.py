"""Repository checks that would otherwise fail only inside a queued GPU job: every shell file parses, every script or
config a file names exists (a moved or deleted file leaves dangling references), and ruff finds nothing (when the
pinned ruff of pyproject.toml is installed: a local venv; the Miles image has none)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OURS = ("configs", "scripts", "prl", "transforms", "tests", "patches")
# scripts/<...>.sh|py|inner and configs/<...>.env, as a path from the repo root (optionally after $PRL_ROOT/)
NAMED = re.compile(r"(?<![\w/.-])(?:\$PRL_ROOT/)?((?:scripts|configs)/[\w./-]+?\.(?:sh|py|inner|env))\b")


def _files(*suffixes: str) -> list[Path]:
    return sorted(p for top in OURS for p in (ROOT / top).rglob("*")
                  if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)


def _shell_files() -> list[Path]:
    bins = [p for p in (ROOT / "scripts").glob("*/bin/*") if p.read_text(errors="replace").startswith("#!/bin/bash")]
    return [*_files(".sh", ".inner", ".env"), *bins]


@pytest.mark.parametrize("path", _shell_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_shell_file_parses(path):
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


# files the README's quick start has the user create (step 2: the site profile)
USER_FILES = {"configs/sites/mysite.env"}


def test_every_named_script_and_config_exists():
    missing = []
    # code and the live READMEs; docs/ is dated records (progress logs, results), which name the files of their day
    for path in [*_files(".sh", ".inner", ".env", ".py", ".md"), ROOT / "README.md", ROOT / "README.zh-CN.md"]:
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for name in NAMED.findall(line):
                if ("<" not in name and "*" not in name and "$" not in name and name not in USER_FILES
                        and not (ROOT / name).exists()):
                    missing.append(f"{path.relative_to(ROOT)}:{line_no}: {name}")
    assert not missing, "\n".join(missing)


def _pinned_ruff() -> str:
    return re.search(r'"ruff==([\d.]+)"', (ROOT / "pyproject.toml").read_text())[1]


def test_ruff_finds_nothing():
    ruff = shutil.which("ruff") or str(Path(sys.executable).with_name("ruff"))
    if not Path(ruff).exists():
        pytest.skip("ruff not installed (the Miles image)")
    version = subprocess.run([ruff, "--version"], capture_output=True, text=True, check=False).stdout.split()[-1]
    if version != _pinned_ruff():
        pytest.skip(f"ruff {version}, not the pinned {_pinned_ruff()} (another default rule set)")
    # The repository's own files only: a user's transforms/<name>.py (untracked) is theirs to lint or not
    tracked = subprocess.run(["git", "ls-files", "prl", "tests", "scripts", "transforms"], cwd=ROOT,
                             capture_output=True, text=True, check=False)
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    files = [f for f in tracked.stdout.split() if f.endswith(".py") and (ROOT / f).exists()]
    result = subprocess.run([ruff, "check", *files], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout
