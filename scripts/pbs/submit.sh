#!/bin/bash
# Submit a chain job under PBS Pro (qsub), from the repo root on a login node; called by scripts/submit.sh for a site
# whose profile sets PRL_SCHEDULER=pbs. Runs scripts/pbs/job.sh on one whole node (chains on several nodes need
# PRL_SCHEDULER=pjm). Site settings (configs/sites/<site>.env):
#   PRL_PBS_QUEUE   queue (QUEUE=<queue> overrides it for one submission, e.g. a debug queue)
#   PRL_PBS_ARGS    further qsub options, e.g. "-W group_list=<group>"
# AFTER=<job id> starts the job once that job ended well (qsub -W depend=afterok). PBS passes variables only through -v
# and takes only ASCII in them. ELAPSE (hh:mm:ss) as for scripts/pjm/submit.sh. When the login node cannot run the
# compute nodes' Python (PRL_SUBMIT_FINDS_SMOKE=0), train_eval looks for its smoke when the job starts.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PRL_ROOT=$PWD PRL_SITE=${PRL_SITE:?set PRL_SITE}
set -a; source "configs/sites/$PRL_SITE.env"; set +a
source scripts/lib/chain.sh
source scripts/lib/submit_chain.sh
after=(); [ -n "${AFTER:-}" ] && after=(-W "depend=afterok:$AFTER")
[ "$nodes" = 1 ] || { echo "scripts/pbs runs one node per job; this chain's arms want $nodes" >&2; exit 1; }
read -ra extra <<< "${PRL_PBS_ARGS:-}"
submit_job qsub -q "${QUEUE:-${PRL_PBS_QUEUE:?set PRL_PBS_QUEUE in configs/sites/$PRL_SITE.env}}" -l select=1 \
  -l walltime="$elapse" -N "${CHAIN//+/_}" -j oe "${after[@]}" "${extra[@]}" -o "$PRL_EXT/logs/" -v "$vars" \
  scripts/pbs/job.sh
