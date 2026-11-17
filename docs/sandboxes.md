# Sandboxes: task images and templates

Every trial runs in its own [E2B](https://e2b.dev) sandbox, created from a template prebuilt for its task. Harbor's
`E2BEnvironment` talks to the sandbox service through the e2b SDK, so nothing here is tied to one E2B-compatible
service; its endpoint and key file are in the site profile (`E2B_API_URL`, `E2B_DOMAIN`, `E2B_API_KEY_FILE`;
`configs/sites/example.env`).

A template belongs to one account and one region. A new account, or a new region, needs its own templates and its own
manifest.

## What you need

- An account on the sandbox service and an API key, saved as the file `E2B_API_KEY_FILE`
  names (chmod 600).
- A container registry the service can pull from, and its credentials (kept outside the repo).
- A machine with Docker and `sudo` for the builds (`linux/amd64`), with the task directories
  (`scripts/setup/prepare_otagent.sh` downloads them).
- Python with `e2b==2.50.0` and `pyqwest==0.10.0` for the registration (the versions the templates were checked with;
  `scripts/setup/install_harbor_deps.sh` installs the same SDK for the jobs).

## 0. Make a snapshot

The arms run on task snapshots: a directory per task set that holds the task directories its templates were built
from, the manifest of those templates and the oracle / no-op results. One for training, one for evaluation:

```bash
python3 scripts/tools/make_task_snapshot.py --tasks-dir $PRL_EXT/data/harbor_tasks \
  --rows $PRL_EXT/data/otagent_rl/train.jsonl --out $PRL_EXT/task_snapshots/otagent728 --tmux --smoke 8
python3 scripts/tools/make_task_snapshot.py --tasks-dir $PRL_EXT/data/harbor_tasks \
  --rows $PRL_EXT/data/ot_tbdev/tbdev.jsonl --out $PRL_EXT/task_snapshots/tbdev70 --tmux
```

Those are the default locations (`PRL_TRAIN_SNAPSHOT`, `PRL_EVAL_SNAPSHOT` in `configs/arms/common_ota.env`; a
site profile may set others). `--tmux` adds an `apt-get install tmux` layer to every task image that lacks it: Terminus-2
drives the shell through tmux, and installing it at the start of each trial costs about 10 s and loses the trial when
the install fails. `--smoke 8` writes the framework smoke's prompts (`smoke.jsonl`). A snapshot is never changed once
its templates are built; a new task set or a new image gets a new directory.

## 1. Build the images

```bash
PRL_IMAGE_TASK_ROOT=$S/tasks \
PRL_IMAGE_TASKS=<task>,<task>,... \
PRL_IMAGE_OUTPUT=<new empty directory> \
PRL_DOCKER_COMMAND='sudo docker' \
python3 scripts/tools/prebuild_task_images.py
```

Only each task's `environment/` directory is sent to Docker (`--platform linux/amd64 --provenance=false --sbom=false`:
a plain single-platform image); tests and solutions are uploaded by Harbor per trial and are not baked in. The output directory gets a
`manifest.json` with one row per task: the local image tag, the context hash of `environment/`, the resolved base-image
digests and the build log. A failed build stops the batch, so that no task silently drops out of an evaluation.

## 2. Push and register

For each task in that manifest: tag the image for your registry, push it, and build a template from it. The
template build is synchronous; mark a row ready only after it returned.

```python
import json, os, subprocess, time
from pathlib import Path
from e2b import Template, default_build_logger

path = Path(os.environ["BUILD_MANIFEST"])            # the manifest.json of step 1
manifest = json.loads(path.read_text())
task = os.environ["TASK"]
row = manifest["tasks"][task]
creds = json.loads(Path(os.environ["REGISTRY_CREDENTIALS_FILE"]).read_text())   # {"username": ..., "password": ...}
image = f"{os.environ['REGISTRY']}/{task}:{manifest['run_id'].lower()}"      # e.g. registry.example.com/project
subprocess.run(["sudo", "docker", "tag", row["image_tag"], image], check=True)
subprocess.run(["sudo", "docker", "push", image], check=True)

api_url, domain = os.environ["E2B_API_URL"], os.environ["E2B_DOMAIN"]           # the site profile's
build = Template.build(
    Template().from_image(image, username=creds["username"], password=creds["password"]),
    name=f"task-{task}-{int(time.time())}", cpu_count=2, memory_mb=2048,
    on_build_logs=default_build_logger(),
    api_key=Path(os.environ["E2B_API_KEY_FILE"]).read_text().strip(), api_url=api_url, domain=domain,
)
row.update(status="ready", template_id=build.template_id, build_id=build.build_id, registry_image=image)
manifest.update(api_url=api_url, domain=domain)
path.write_text(json.dumps(manifest, indent=2) + "\n")
```

Log in to the registry with `docker login --password-stdin` and keep the Docker config outside the repo. The template
fixes the sandbox's resources (2 CPU, 2 GB here); `HARBOR_OVERRIDE_CPUS` / `HARBOR_OVERRIDE_MEMORY_MB` in
`configs/arms/common_ota.env` only record the same values for Harbor.

## 3. Put the manifest into the snapshot

Copy the registered manifest to `$S/manifest.json` (`$S` = the snapshot): the training arms read
`$PRL_TRAIN_SNAPSHOT/manifest.json`, the evaluation arm `$PRL_EVAL_SNAPSHOT/manifest.json`. Before a sandbox is
created,
`prl/sandbox/template_manifest.py` checks that the manifest belongs to the configured `E2B_API_URL`, that the task's
row is `ready`, and that the task's `environment/` still has the context hash it was built from; a changed task
directory fails there, and needs a rebuild.

## 4. Check them

- `python3 scripts/checks/sandbox_smoke.py`: the sandbox contract Harbor relies on (files, exec, tmux), without
  a model.
- `CHAIN=task_check PRL_TEMPLATE_MANIFEST=$S/manifest.json PRL_TASK_ROOT=$S/tasks PRL_TASK_CHECK_OUTPUT=$S/contract
  bash scripts/submit.sh`: each task's official oracle must score 1 and a no-op agent 0 on its template
  (`scripts/chain/task_smoke.sh`). The framework smoke's preflight requires these results for its 8 tasks
  (`PRL_TASK_PROMPTS=$S/smoke.jsonl` checks only those).
