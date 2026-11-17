"""Image identity must change with task content and exclude outside-context files."""

from __future__ import annotations

import shutil

import pytest

from scripts.tools.prebuild_task_images import context_hash, context_inventory


def test_context_identity_is_portable_and_tracks_nested_content(tmp_path):
    original = tmp_path / "original"
    (original / "seeds").mkdir(parents=True)
    (original / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY seeds/ /workspace/\n")
    (original / "seeds/input").write_text("first")
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    initial = context_hash(context_inventory(original))
    assert context_hash(context_inventory(copied)) == initial
    (copied / "seeds/input").write_text("second")
    assert context_hash(context_inventory(copied)) != initial


def test_context_identity_tracks_executable_permissions(tmp_path):
    script = tmp_path / "setup.sh"
    script.write_text("#!/bin/sh\ntrue\n")
    script.chmod(0o644)
    initial = context_hash(context_inventory(tmp_path))
    script.chmod(0o755)
    assert context_hash(context_inventory(tmp_path)) != initial


def test_context_rejects_external_symlinks(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    (tmp_path / "outside").write_text("not part of the build context")
    (context / "link").symlink_to("../outside")
    with pytest.raises(ValueError, match="Symlink leaves"):
        context_inventory(context)
