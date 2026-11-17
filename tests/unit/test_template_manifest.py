"""A stale image or wrong regional template must fail before sandbox creation."""
from __future__ import annotations

import json

import pytest

from prl.sandbox.template_manifest import context_hash, context_inventory, resolve_template


def test_template_identity_and_fail_closed(tmp_path):
    task = tmp_path / "task"
    context = task / "environment"
    context.mkdir(parents=True)
    (context / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    manifest = tmp_path / "manifest.json"
    row = {"status": "ready", "template_id": "template-a", "context_hash": context_hash(context_inventory(context))}
    data = {"api_url": "https://region.example", "tasks": {"task": row}}
    manifest.write_text(json.dumps(data))
    assert resolve_template(str(manifest), task, "https://region.example") == "template-a"
    with pytest.raises(ValueError, match="different E2B endpoint"):
        resolve_template(str(manifest), task, "https://another.example")
    row["status"] = "building"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not ready"):
        resolve_template(str(manifest), task, "https://region.example")
    row["status"] = "ready"
    manifest.write_text(json.dumps(data))
    (context / "Dockerfile").write_text("FROM debian:bookworm\n")
    with pytest.raises(ValueError, match="context changed"):
        resolve_template(str(manifest), task, "https://region.example")
