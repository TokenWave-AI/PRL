#!/bin/bash
# The sandboxes of the quick start in one command, on a linux/amd64 machine with Docker and internet: the tasks
# (scripts/setup/prepare_otagent.sh, skipped when there), the training and TB-dev snapshots
# (scripts/tools/make_task_snapshot.py: /output and tmux layers, the smoke's 8 prompts in the training one), an image per task
# (scripts/tools/prebuild_task_images.py), pushed to your registry and registered as a template on the sandbox service
# (scripts/tools/register_templates.py), and each snapshot's manifest.json. From the repo root:
#   PRL_SITE=<site> bash scripts/setup/build_sandboxes.sh
#   PRL_SITE=<site> PRL_SANDBOX_SMOKE_ONLY=1 bash scripts/setup/build_sandboxes.sh   # only the smoke's 8 tasks: a first try
# Settings from the site profile (configs/sites/example.env) or the environment: E2B_API_URL, E2B_DOMAIN,
# E2B_API_KEY_FILE, PRL_REGISTRY, PRL_REGISTRY_CREDENTIALS_FILE (log Docker in to the registry as well),
# PRL_DOCKER_COMMAND (default "sudo docker"); PRL_EXT may be set to a directory on this machine. Batches of
# PRL_SANDBOX_BATCH tasks (default 50) are built, pushed and registered in turn, their local images removed, so the
# disk holds one batch. Run it again after a failure: it goes on with the tasks that have no ready template.
# A Python >= 3.10 (PRL_SANDBOX_PYTHON, default python3) gets a venv in $PRL_EXT/sandbox_venv with the e2b SDK, hf and
# pyarrow. When this machine is not the cluster, copy $PRL_EXT/task_snapshots to the cluster's PRL_EXT afterwards.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PRL_ROOT=$PWD
if [ -n "${PRL_SITE:-}" ]; then set -a; source "configs/sites/$PRL_SITE.env"; set +a; fi
: "${PRL_EXT:?set PRL_SITE or PRL_EXT}" "${E2B_API_URL:?}" "${E2B_DOMAIN:?}" "${E2B_API_KEY_FILE:?}"
: "${PRL_REGISTRY:?set PRL_REGISTRY (registry/project the sandbox service pulls from) in the site profile}"
: "${PRL_REGISTRY_CREDENTIALS_FILE:?set PRL_REGISTRY_CREDENTIALS_FILE in the site profile}"
export PRL_EXT E2B_API_URL E2B_DOMAIN E2B_API_KEY_FILE PRL_REGISTRY PRL_REGISTRY_CREDENTIALS_FILE
export PRL_DOCKER_COMMAND=${PRL_DOCKER_COMMAND:-sudo docker} PRL_IMAGE_PRUNE=1
# this machine's own cache (the profile's HF_HOME may be a cluster path): the datasets go to $PRL_EXT/data anyway
export PRL_SCRATCH=$PRL_EXT/sandbox_scratch HF_HOME=$PRL_EXT/sandbox_scratch/hf
mkdir -p "$PRL_SCRATCH"
# the snapshots the arms read (configs/arms/common_ota.env): a site profile may name others
train=${PRL_TRAIN_SNAPSHOT:-$PRL_EXT/task_snapshots/otagent728} eval=${PRL_EVAL_SNAPSHOT:-$PRL_EXT/task_snapshots/tbdev70}

venv=$PRL_EXT/sandbox_venv
if [ ! -x "$venv/bin/hf" ]; then
  "${PRL_SANDBOX_PYTHON:-python3}" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 needed"'
  "${PRL_SANDBOX_PYTHON:-python3}" -m venv "$venv"
  "$venv/bin/pip" install -q e2b==2.50.0 pyqwest==0.10.0 huggingface_hub pyarrow
fi
py=$venv/bin/python

echo "== tasks -> $PRL_EXT/data"
if [ ! -s "$PRL_EXT/data/otagent_rl/train.jsonl" ] || [ ! -s "$PRL_EXT/data/ot_tbdev/tbdev.jsonl" ]; then
  MILES_IMAGE=none PREP_PYTHON=$py HF_CLI=$venv/bin/hf MODELS= bash scripts/setup/prepare_otagent.sh
fi

echo "== snapshots"
[ -d "$train" ] || "$py" scripts/tools/make_task_snapshot.py --tasks-dir "$PRL_EXT/data/harbor_tasks" \
  --rows "$PRL_EXT/data/otagent_rl/train.jsonl" --out "$train" --output-dir --tmux --smoke 8
[ "${PRL_SANDBOX_SMOKE_ONLY:-0}" = 1 ] || [ -d "$eval" ] || "$py" scripts/tools/make_task_snapshot.py \
  --tasks-dir "$PRL_EXT/data/harbor_tasks" --rows "$PRL_EXT/data/ot_tbdev/tbdev.jsonl" --out "$eval" --tmux

# todo SNAPSHOT [ROWS]: the snapshot's tasks (or only ROWS' tasks) without a ready template, one per line
todo() {
  "$py" - "$@" <<'PY'
import json, sys
from pathlib import Path
snapshot = Path(sys.argv[1])
if len(sys.argv) > 2:
    tasks = [json.loads(line)["metadata"]["instance_id"] for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
else:
    tasks = sorted(p.name for p in (snapshot / "tasks").iterdir() if p.is_dir())
manifest = snapshot / "manifest.json"
ready = {t for t, r in json.loads(manifest.read_text())["tasks"].items() if r["status"] == "ready"} if manifest.exists() else set()
print("\n".join(t for t in tasks if t not in ready))
PY
}

snapshots=("$train"); [ "${PRL_SANDBOX_SMOKE_ONLY:-0}" = 1 ] || snapshots+=("$eval")
for s in "${snapshots[@]}"; do
  rows=(); [ "${PRL_SANDBOX_SMOKE_ONLY:-0}" = 1 ] && rows=("$s/smoke.jsonl")
  mapfile -t left < <(todo "$s" "${rows[@]}" | sed '/^$/d')
  echo "== $s: ${#left[@]} tasks without a template"
  n=${PRL_SANDBOX_BATCH:-50}
  for ((i = 0; i < ${#left[@]}; i += n)); do
    batch=("${left[@]:i:n}") out=$PRL_EXT/task_builds/$(basename "$s")/$(date -u +%Y%m%dT%H%M%S)
    mkdir -p "$(dirname "$out")"
    echo "   batch $((i / n + 1)): ${#batch[@]} tasks, build record $out"
    PRL_IMAGE_TASK_ROOT=$s/tasks PRL_IMAGE_TASKS=$(IFS=,; echo "${batch[*]}") PRL_IMAGE_OUTPUT=$out \
      "$py" scripts/tools/prebuild_task_images.py > "$out.log"
    PRL_BUILD_MANIFEST=$out/manifest.json PRL_SNAPSHOT_MANIFEST=$s/manifest.json "$py" scripts/tools/register_templates.py
  done
  left=$(todo "$s" "${rows[@]}" | sed '/^$/d' | wc -l)
  [ "$left" = 0 ] || { echo "$s: $left tasks still have no template" >&2; exit 1; }
done
echo "== done: ${snapshots[*]} have their templates (manifest.json)"
echo "   on another machine than the cluster: copy $PRL_EXT/task_snapshots to the cluster's PRL_EXT"
