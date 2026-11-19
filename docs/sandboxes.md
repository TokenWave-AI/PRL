# Sandboxes: task images and templates

Every trial runs in its own [E2B](https://e2b.dev) sandbox, created from a template prebuilt for its task. Harbor's
`E2BEnvironment` talks to the sandbox service through the e2b SDK, so nothing here is tied to one E2B-compatible
service; its endpoint and key file are in the site profile (`E2B_API_URL`, `E2B_DOMAIN`, `E2B_API_KEY_FILE`;
`configs/sites/example.env`). A template belongs to one account and one region: a new account, or a new region, needs
its own templates.

## One command

On a linux/amd64 machine with Docker (buildx), `sudo`, internet and Python ≥ 3.10, from the repository root:

```bash
PRL_SITE=<site> bash scripts/setup/build_sandboxes.sh
```

Besides the sandbox endpoint, it reads from the site profile the registry the images go to (`PRL_REGISTRY`) and the
credentials the service pulls them with (`PRL_REGISTRY_CREDENTIALS_FILE`, `{"username": ..., "password": ...}`,
chmod 600). The script calls `sudo docker`, so log root's Docker in to that registry (`sudo docker login
--password-stdin`), or set `PRL_DOCKER_COMMAND="sudo docker --config <dir>"` for a Docker config logged in elsewhere (a directory of
your own: buildx keeps its current builder there too).
On another machine than the cluster, set `PRL_EXT` to a directory there, with the key files where the profile names them
(or set `E2B_API_KEY_FILE` and `PRL_REGISTRY_CREDENTIALS_FILE` too), and copy `$PRL_EXT/task_snapshots` to the cluster's
`PRL_EXT` afterwards. `PRL_SANDBOX_SMOKE_ONLY=1` builds only the 8 tasks of the framework smoke (a few minutes; all 798
take hours). It stops at the first task that fails; run it again to go on with the tasks that have no template yet.
Docker keeps its build cache (`docker builder prune` frees it).

What it does, with the tools it calls (each usable alone):

1. The tasks (`scripts/setup/prepare_otagent.sh`), when `$PRL_EXT/data` lacks them.
2. Two **snapshots**, `$PRL_EXT/task_snapshots/otagent728` (training, with the smoke's 8 prompts in `smoke.jsonl`) and
   `tbdev70` (`scripts/tools/make_task_snapshot.py`): copies of the task directories the templates are built from. The
   training tasks' solutions and tests write to `/output`, which their images lack: `--output-dir` adds it. The arms
   read the snapshots as `PRL_TRAIN_SNAPSHOT` and `PRL_EVAL_SNAPSHOT` (`configs/arms/common_ota.env`; a site profile
   may name others). `--tmux` adds a tmux layer to every task image that lacks it: Terminus-2 drives the shell through
   tmux, and installing it at the start of each trial costs about 10 s and loses the trial when the install fails.
3. An **image** per task from its `environment/` directory (`scripts/tools/prebuild_task_images.py`: linux/amd64, no
   provenance or SBOM attestation; tests and solutions are uploaded by Harbor per trial, not baked in).
4. The image **pushed** and a **template** built from it on the sandbox service, 2 CPUs and 2 GB
   (`scripts/tools/register_templates.py`), batch by batch, the local images removed after each batch.
5. Each snapshot's **`manifest.json`**: task → ready template and the context hash of the task's `environment/`.

A snapshot is never changed once its templates are built: before a sandbox is created, `prl/sandbox/template_manifest.py`
checks that the manifest belongs to the configured `E2B_API_URL`, that the task's row is ready, and that the task's
`environment/` still has the hash it was built from. A new task set or a changed image gets a new snapshot directory.

## Checks

- `python3 scripts/checks/sandbox_smoke.py`: the sandbox contract Harbor relies on (files, exec, tmux), without a model.
- The official solution must score 1 and a no-op agent 0 on each task's template (`scripts/chain/task_smoke.sh`). For the
  smoke's 8 tasks, into `<training snapshot>/contract`: `PRL_SITE=<site> CHAIN=task_check bash scripts/submit.sh` (the
  first framework smoke does it by itself when it was not done); for a whole snapshot:
  `PRL_SITE=<site> CHAIN=task_check PRL_TEMPLATE_MANIFEST=$S/manifest.json PRL_TASK_ROOT=$S/tasks bash scripts/submit.sh`.
