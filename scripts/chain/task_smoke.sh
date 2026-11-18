#!/bin/bash
# Chain step task_check (scripts/lib/chain.sh): official oracle / nop Harbor trials against prebuilt task templates
# (scripts/checks/task_smoke.py: oracle must score 1, nop 0, no infrastructure exception), before a new template
# snapshot is used. No model. PRL_TEMPLATE_MANIFEST and PRL_TASK_ROOT name the snapshot S
# (scripts/tools/make_task_snapshot.py), e.g.
#   CHAIN=task_check PRL_TEMPLATE_MANIFEST=$S/manifest.json PRL_TASK_ROOT=$S/tasks bash scripts/submit.sh
# PRL_TASK_PROMPTS (a prompt JSONL) checks only its tasks; the smoke preflight reuses <manifest dir>/contract for
# its 8 tasks, written by
#   CHAIN=task_check+smoke_grpo PRL_TEMPLATE_MANIFEST=$S/manifest.json PRL_TASK_ROOT=$S/tasks \
#     PRL_TASK_PROMPTS=$S/smoke.jsonl PRL_TASK_CHECK_OUTPUT=$S/contract ...
# The sandbox endpoint and key file are the site profile's; the Harbor dependencies the training arms'
# (ota_train_smoke).
set -euo pipefail
export PRL_ROOT=${PRL_ROOT:-$PWD} PRL_SITE=${PRL_SITE:?set PRL_SITE (configs/sites)}
# the snapshot to check, kept before the arm (which names the training snapshot) is sourced
manifest=${PRL_TEMPLATE_MANIFEST:?the template manifest to check} tasks=${PRL_TASK_ROOT:?the task snapshot it was built from}
set -a
source "$PRL_ROOT/configs/sites/$PRL_SITE.env"
source "$PRL_ROOT/configs/arms/common_ota_single_node.env"
source "$PRL_ROOT/configs/arms/ota_train_smoke.env"
set +a
export PRL_TASK_CHECK_OUTPUT=${PRL_TASK_CHECK_OUTPUT:-$PRL_EXT/task_checks/${JOB_ID:-$$}}
export LITELLM_LOCAL_MODEL_COST_MAP=True HARBOR_TELEMETRY=0
unset E2B_API_KEY E2B_SANDBOX_URL
# as the training container (scripts/lib/env.sh in_miles): the repo, PRL_EXT and PRL_EXTRA_BINDS (e.g. where the key
# files are)
binds=(-B "$PRL_ROOT" -B "$PRL_EXT"); for d in ${PRL_EXTRA_BINDS:-}; do if [ -e "$d" ]; then binds+=(-B "$d"); fi; done
apptainer exec "${binds[@]}" \
  --env "PYTHONPATH=$PRL_ROOT:${PRL_EXTRA_PYTHONPATH:-}:$PRL_EXT/pydeps_harbor" \
  --env "E2B_API_KEY_FILE=$E2B_API_KEY_FILE,E2B_API_URL=$E2B_API_URL,E2B_DOMAIN=$E2B_DOMAIN" \
  --env "PRL_TEMPLATE_MANIFEST=$manifest,PRL_TASK_ROOT=$tasks,PRL_TASK_PROMPTS=${PRL_TASK_PROMPTS:-}" \
  --env "PRL_TASK_CHECK_OUTPUT=$PRL_TASK_CHECK_OUTPUT,LITELLM_LOCAL_MODEL_COST_MAP=True,HARBOR_TELEMETRY=0" \
  "$MILES_IMAGE" /opt/sglang/bin/python3 "$PRL_ROOT/scripts/checks/task_smoke.py"
