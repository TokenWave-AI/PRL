"""Build task environments into single-platform images for sandbox template registration.

Run on a build host with Docker, before uploading images or starting Harbor trials.
Only each task's environment/ directory is sent to Docker. Outputs include context
fingerprints, exact commands, logs, resolved base-image digests and image manifests.
This tool does not push images, create templates, or run models. A failed build is
recorded and terminates the batch; it must not silently remove an evaluation task.

Configuration:
    PRL_IMAGE_TASK_ROOT: directory containing Harbor task directories.
    PRL_IMAGE_TASKS: required comma-separated task directory names.
    PRL_IMAGE_OUTPUT: required output directory for a new build run.
    PRL_IMAGE_PREFIX: image repository name, default harbor-task.
    PRL_DOCKER_COMMAND: command prefix, default docker (e.g. sudo docker).
    PRL_IMAGE_BUILD_TIMEOUT: per-image timeout in seconds, default 1800.

A template is built from a prebuilt image: linux/amd64, single-platform, without provenance/SBOM
attestations (an attestation manifest list is not a plain image).
BuildKit's separate metadata JSON still records the resolved base-image digests.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Preserve direct script invocation without requiring an editable install.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prl.sandbox.template_manifest import context_hash, context_inventory


def main() -> None:
    task_root = Path(os.environ["PRL_IMAGE_TASK_ROOT"]).resolve(strict=True)
    tasks = list(dict.fromkeys(item.strip() for item in os.environ["PRL_IMAGE_TASKS"].split(",") if item.strip()))
    if not tasks or any(Path(task).name != task or task in (".", "..") for task in tasks):
        raise ValueError("PRL_IMAGE_TASKS must contain task directory names")
    output = Path(os.environ["PRL_IMAGE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    docker = shlex.split(os.getenv("PRL_DOCKER_COMMAND", "docker"))
    prefix = os.getenv("PRL_IMAGE_PREFIX", "harbor-task")
    timeout = int(os.getenv("PRL_IMAGE_BUILD_TIMEOUT", "1800"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    manifest: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "platform": "linux/amd64",
        "provenance_attestation": False, "sbom_attestation": False,
        "push": False, "tasks": {},
    }

    def save() -> None:
        temporary = output / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    save()
    for task in tasks:
        context = (task_root / task / "environment").resolve(strict=True)
        if not context.is_relative_to(task_root) or not (context / "Dockerfile").is_file():
            raise ValueError(f"Missing or invalid Dockerfile for {task}")
        inventory = context_inventory(context)
        digest = context_hash(inventory)
        directory = output / task
        directory.mkdir()
        (directory / "context.json").write_text(json.dumps(inventory, indent=2) + "\n")
        # Unique tags avoid overwriting a published image if a mutable base tag
        # resolves differently while the task's Dockerfile stays unchanged.
        tag = f"{prefix}:{digest[:20]}-{run_id}"
        metadata_path = directory / "build_metadata.json"
        command = [
            *docker, "buildx", "build", "--platform", "linux/amd64", "--provenance=false", "--sbom=false",
            "--load", "--progress=plain", "--metadata-file", str(metadata_path.resolve()),
            "--tag", tag, str(context),
        ]
        row: dict[str, Any] = {
            "context": str(context), "context_hash": digest, "image_tag": tag,
            "command": command, "status": "building", "template_id": None,
        }
        manifest["tasks"][task] = row
        save()
        try:
            with (directory / "build.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
            if context_hash(context_inventory(context)) != digest:
                raise RuntimeError("Image context changed during build")
            metadata = json.loads(metadata_path.read_text())
            descriptor = metadata["containerimage.descriptor"]
            if descriptor.get("platform") != {"architecture": "amd64", "os": "linux"}:
                raise ValueError("Expected a single linux/amd64 image manifest")
            row.update(status="built", image_digest=metadata["containerimage.digest"])
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            save()
        print(json.dumps({"task": task, "status": row["status"], "image_tag": tag, "image_digest": row["image_digest"]}))


if __name__ == "__main__":
    main()
