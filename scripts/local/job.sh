#!/bin/bash
# A chain job on this machine: the steps in PRL_CHAIN (scripts/lib/chain.sh lists them) on the site PRL_SITE. Started by
# scripts/local/submit.sh (through scripts/submit.sh), which passes the job id and the chain's variables as
# VAR=value,... (scripts/lib/submit_chain.sh):
#   bash scripts/local/job.sh <job id> <VAR=value,...>
# Detaches into the background (log $PRL_EXT/logs/local_<job id>.out) unless FOREGROUND=1.
set -euo pipefail
cd "$(dirname "$0")/../.."
job=${1:?job id} vars=${2:?the chain variables}
IFS=, read -ra assignments <<< "$vars"
export "${assignments[@]}"
export PRL_ROOT=$PWD PRL_SITE=${PRL_SITE:?the variables name no PRL_SITE} JOB_ID=$job
set -a; source "configs/sites/$PRL_SITE.env"; set +a
mkdir -p "$PRL_EXT/logs" "$PRL_EXT/locks"
if [ "${FOREGROUND:-0}" != 1 ] && [ -z "${PRL_LOCAL_DETACHED:-}" ]; then
  log=$PRL_EXT/logs/local_$job.out
  PRL_LOCAL_DETACHED=1 setsid nohup bash "$0" "$job" "$vars" > "$log" 2>&1 < /dev/null &
  echo "local job $job started in the background (pid $!); log: $log"
  exit 0
fi
# Two jobs on one machine would share its GPUs: the second stops here
exec 9> "$PRL_EXT/locks/local_$(hostname -s).lock"
flock -n 9 || { echo "another local job runs on $(hostname -s) (lock $PRL_EXT/locks/local_$(hostname -s).lock)" >&2; exit 1; }
source scripts/lib/chain.sh
run_chain
