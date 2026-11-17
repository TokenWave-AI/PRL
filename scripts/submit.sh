#!/bin/bash
# Submit a chain job (scripts/lib/chain.sh) on the cluster PRL_SITE names: its profile configs/sites/<site>.env
# (start from configs/sites/example.env) says which scheduler it has (PRL_SCHEDULER: pjm or pbs), and that
# scheduler's wrapper scripts/<scheduler>/submit.sh submits the job. From the repo root on a login node:
#   PRL_SITE=<site> CHAIN=train_eval ARM=ota_train_transform TRANSFORM=atlas PRL_RUN_ID=my_atlas bash scripts/submit.sh
#   PRL_SITE=<site> CHAIN=smoke_grpo TRANSFORM=atlas bash scripts/submit.sh          # the framework smoke
#   PRL_SITE=<site> CHAIN=eval_hf MODEL_HF_DIR=$PRL_EXT/ckpt/hf/<run>_iter_... bash scripts/submit.sh
# The steps and the variables they read: scripts/lib/chain.sh, scripts/lib/submit_chain.sh. DRY_RUN=1 prints the
# scheduler command and submits nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PRL_SITE:?set PRL_SITE to a cluster with configs/sites/<site>.env (see configs/sites/example.env)}"
[ -f "configs/sites/$PRL_SITE.env" ] || { echo "no configs/sites/$PRL_SITE.env" >&2; exit 1; }
scheduler=$(export PRL_ROOT=$PWD; set -a; source "configs/sites/$PRL_SITE.env"; echo "${PRL_SCHEDULER:-}")
[ -f "scripts/${scheduler:-none}/submit.sh" ] || {
  echo "configs/sites/$PRL_SITE.env: PRL_SCHEDULER=${scheduler:-(unset)}; supported: pjm, pbs" >&2; exit 1; }
exec bash "scripts/$scheduler/submit.sh" "$@"
