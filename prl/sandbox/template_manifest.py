"""Validate prebuilt task templates before Harbor creates a sandbox.

Used by the image builder and rollout.harbor_agent; a manifest binds each task's
complete environment context to a ready template in one explicit E2B endpoint.
No dataset-specific assumptions or credentials belong here. Configuration arrives
as arguments from HARBOR_ENV_KWARGS (prl_template_manifest); no env reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


def context_inventory(root: Path) -> list[dict[str, Any]]:
    """Fingerprint all context entries conservatively, including Dockerfile/ignore rules.

    Include entries excluded by .dockerignore too: an extra rebuild is preferable
    to accidentally reusing a template when Docker's context interpretation changes.
    Reject links leaving the context rather than hashing an external target by accident.
    """
    root = root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        entry: dict[str, Any] = {"path": path.relative_to(root).as_posix(), "mode": stat.S_IMODE(mode)}
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Symlink leaves image context: {entry['path']}")
            entry.update(kind="symlink", target=os.readlink(path))
        elif path.is_dir():
            entry["kind"] = "directory"
        elif path.is_file():
            entry.update(kind="file", bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            raise ValueError(f"Unsupported image context entry: {entry['path']}")
        records.append(entry)
    return records


def context_hash(inventory: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_template(manifest_path: str, task_path: Path, api_url: str) -> str:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["api_url"].rstrip("/") != api_url.rstrip("/"):
        raise ValueError("Template manifest belongs to a different E2B endpoint")
    row = manifest["tasks"][task_path.name]
    if row["status"] != "ready" or not row.get("template_id"):
        raise ValueError(f"Template is not ready for {task_path.name}")
    if row["context_hash"] != context_hash(context_inventory(task_path / "environment")):
        raise ValueError(f"Image context changed for {task_path.name}; rebuild its template")
    return row["template_id"]
