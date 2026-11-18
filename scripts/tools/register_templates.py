"""Push built task images and register each as a sandbox template; write the snapshot's template manifest.

Runs after scripts/tools/prebuild_task_images.py on the same Docker machine (scripts/setup/build_sandboxes.sh runs
both, batch by batch). For every task of the build manifest whose image was built: tag it for your registry, push it,
build a template from it on the E2B-compatible sandbox service (the e2b SDK's Template.build, synchronous), and add
the task to the snapshot's manifest as ready. The snapshot's manifest is what the arms read
(PRL_TRAIN_SNAPSHOT / PRL_EVAL_SNAPSHOT + /manifest.json; prl/sandbox/template_manifest.py checks a row before a sandbox
is created); it is written after every template, so a rerun skips the tasks already ready with the same context hash.
A task that still fails after three tries is reported and the tool exits 1, so no task silently drops out.

Configuration (environment):
    PRL_BUILD_MANIFEST       manifest.json of prebuild_task_images.py (rows with status "built")
    PRL_SNAPSHOT_MANIFEST    the snapshot's manifest.json, created or updated
    PRL_REGISTRY             where the images go, e.g. registry.example.com/project (the service pulls from it)
    PRL_REGISTRY_CREDENTIALS_FILE  {"username": ..., "password": ...} (chmod 600): the service pulls with them; log
                             Docker in to the registry too (docker login --password-stdin)
    E2B_API_URL, E2B_DOMAIN, E2B_API_KEY_FILE   the sandbox service, as in the site profile
    PRL_DOCKER_COMMAND       default docker (e.g. sudo docker)
    PRL_IMAGE_PRUNE          1: remove the local images of a task once its template is ready (default 0)
    PRL_TEMPLATE_WORKERS     tasks registered at once (default 8)
    PRL_TEMPLATE_CPUS, PRL_TEMPLATE_MEMORY_MB  the sandbox's resources (default 2, 2048; HARBOR_OVERRIDE_CPUS /
                             HARBOR_OVERRIDE_MEMORY_MB in configs/arms/common_ota.env record the same for Harbor)
    PRL_TEMPLATE_RETRY_WAIT  seconds before the second try (default 30; the third waits twice as long)
Needs e2b==2.50.0 and pyqwest==0.10.0 (the versions the templates were checked with).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

TRIES = 3


def main() -> None:
    from e2b import Template

    build = json.loads(Path(os.environ["PRL_BUILD_MANIFEST"]).read_text())
    target = Path(os.environ["PRL_SNAPSHOT_MANIFEST"])
    registry = os.environ["PRL_REGISTRY"].rstrip("/")
    creds = json.loads(Path(os.environ["PRL_REGISTRY_CREDENTIALS_FILE"]).read_text())
    api = {"api_key": Path(os.environ["E2B_API_KEY_FILE"]).read_text().strip(),
           "api_url": os.environ["E2B_API_URL"], "domain": os.environ["E2B_DOMAIN"]}
    docker = shlex.split(os.getenv("PRL_DOCKER_COMMAND", "docker"))
    prune = os.getenv("PRL_IMAGE_PRUNE", "0") == "1"
    cpus, memory = int(os.getenv("PRL_TEMPLATE_CPUS", "2")), int(os.getenv("PRL_TEMPLATE_MEMORY_MB", "2048"))

    manifest: dict[str, Any] = json.loads(target.read_text()) if target.exists() else {
        "schema_version": 1, "api_url": api["api_url"], "domain": api["domain"], "builds": [], "tasks": {}}
    if manifest["api_url"].rstrip("/") != api["api_url"].rstrip("/"):
        raise SystemExit(f"{target} belongs to {manifest['api_url']}, not {api['api_url']}: a template belongs to one"
                         " account and endpoint, so another one needs its own snapshot")
    if build["run_id"] not in manifest["builds"]:
        manifest["builds"].append(build["run_id"])
    lock = threading.Lock()

    def save() -> None:
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(target)

    todo = [(task, row) for task, row in build["tasks"].items() if row["status"] == "built" and not (
        manifest["tasks"].get(task, {}).get("status") == "ready"
        and manifest["tasks"][task]["context_hash"] == row["context_hash"])]
    print(f"[register] {len(todo)} of {len(build['tasks'])} tasks to register ({target})", flush=True)

    def register(task: str, row: dict[str, Any]) -> str | None:
        image = f"{registry}/{task.lower()}:{build['run_id'].lower()}"
        error = ""
        for attempt in range(TRIES):
            try:
                subprocess.run([*docker, "tag", row["image_tag"], image], check=True, capture_output=True)
                subprocess.run([*docker, "push", image], check=True, capture_output=True)
                template = Template.build(
                    Template().from_image(image, username=creds["username"], password=creds["password"]),
                    name=f"task-{task.lower()}-{int(time.time())}", cpu_count=cpus, memory_mb=memory, **api)
                break
            except Exception as exc:  # noqa: BLE001 - a push or build error is tried again, then reported
                out = getattr(exc, "stderr", None)
                error = f"{type(exc).__name__}: {(out.decode(errors='replace') if out else exc)}"[:500]
                time.sleep(float(os.getenv("PRL_TEMPLATE_RETRY_WAIT", "30")) * (attempt + 1))
        else:
            return error
        with lock:
            manifest["tasks"][task] = {
                "status": "ready", "template_id": template.template_id, "build_id": template.build_id,
                "context_hash": row["context_hash"], "registry_image": image, "image_digest": row.get("image_digest"),
                "build_run": build["run_id"]}
            save()
            ready = sum(r["status"] == "ready" for r in manifest["tasks"].values())
        print(f"[register] {task}: template {template.template_id} ({ready} ready in the snapshot)", flush=True)
        if prune:
            subprocess.run([*docker, "rmi", row["image_tag"], image], check=False, capture_output=True)
        return None

    with ThreadPoolExecutor(int(os.getenv("PRL_TEMPLATE_WORKERS", "8"))) as pool:
        errors = dict(zip([t for t, _ in todo], pool.map(lambda item: register(*item), todo)))
    save()
    failed = {task: error for task, error in errors.items() if error}
    for task, error in failed.items():
        print(f"[register] FAILED {task}: {error}", file=sys.stderr)
    print(f"[register] done: {len(todo) - len(failed)} registered, {len(failed)} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
