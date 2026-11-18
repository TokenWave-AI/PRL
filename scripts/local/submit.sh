#!/bin/bash
# Run a chain job (scripts/lib/chain.sh) on this machine, without a scheduler; called by scripts/submit.sh for a site
# whose profile sets PRL_SCHEDULER=local (one machine with the GPUs an arm asks for, e.g. a cloud GPU server or a node
# you are logged in to). The job runs in the background, so it survives the terminal; its log is
# $PRL_EXT/logs/local_<job id>.out. FOREGROUND=1 runs it in this shell instead, e.g. inside an allocation of another
# scheduler: sbatch -N 1 --gpus 4 --wrap "PRL_SITE=<site> FOREGROUND=1 bash scripts/quickstart.sh".
# One job per machine at a time (a second one stops at its start); AFTER and ELAPSE do not apply: put the steps in one
# chain (CHAIN=a+b), which has no time limit here. DRY_RUN=1 prints the command and starts nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PRL_ROOT=$PWD PRL_SITE=${PRL_SITE:?set PRL_SITE}
set -a; source "configs/sites/$PRL_SITE.env"; set +a
source scripts/lib/chain.sh
source scripts/lib/submit_chain.sh
[ "$nodes" = 1 ] || { echo "scripts/local runs on one machine; this chain's arms want $nodes nodes" >&2; exit 1; }
[ -z "${AFTER:-}" ] || { echo "AFTER does not apply to a local job: put the steps in one chain (CHAIN=a+b)" >&2; exit 1; }
submit_job bash scripts/local/job.sh "local$(date +%Y%m%d%H%M%S)" "$vars"
