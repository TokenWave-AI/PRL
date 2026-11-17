#!/bin/bash
# Submit a chain job under Fujitsu PJM (pjsub), from the repo root on a login node; called by scripts/submit.sh for a
# site whose profile sets PRL_SCHEDULER=pjm. Runs scripts/pjm/job.sh on as many nodes as the chain's arms ask for
# (their NNODES). Site settings (configs/sites/<site>.env):
#   PRL_PJM_RSCGRP          resource group (pjsub -L rscgrp=...); RESERVATION=<id> submits to a reservation instead
#   PRL_PJM_ARGS            further pjsub options, e.g. "-L jobenv=singularity"
#   PRL_PJM_MULTINODE_HOURS longest elapse of a job on more than one node, if the group has a shorter limit for them
# ELAPSE (hh:mm:ss) overrides the chain's default (168 h with train_eval, 8 h with a smoke or an evaluation, else 2 h).
# train_eval without a smoke step uses the newest accepted smoke whose sources match these (scripts/checks/find_smoke.py).
# Records: $PRL_EXT/submissions/<site>_<run id, or chain_model_date>/.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PRL_ROOT=$PWD PRL_SITE=${PRL_SITE:?set PRL_SITE}
set -a; source "configs/sites/$PRL_SITE.env"; set +a
source scripts/lib/chain.sh
source scripts/lib/submit_chain.sh
if [ "$nodes" -gt 1 ] && [ -n "${PRL_PJM_MULTINODE_HOURS:-}" ] && [ "${elapse%%:*}" -gt "$PRL_PJM_MULTINODE_HOURS" ]; then
  elapse=$PRL_PJM_MULTINODE_HOURS:00:00
fi
if [ -n "${RESERVATION:-}" ]; then where=(-r "$RESERVATION" -L "node=$nodes,elapse=$elapse")
else where=(-L "rscgrp=${PRL_PJM_RSCGRP:?set PRL_PJM_RSCGRP in configs/sites/$PRL_SITE.env},node=$nodes,elapse=$elapse"); fi
read -ra extra <<< "${PRL_PJM_ARGS:-}"
submit_job pjsub "${where[@]}" "${extra[@]}" -N "${CHAIN//+/_}" -o "$PRL_EXT/logs/pjm_%n_%j.out" -x "$vars" \
  scripts/pjm/job.sh
