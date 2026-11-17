#!/bin/bash
#PBS -j oe
# A chain job under PBS Pro: the steps in PRL_CHAIN (scripts/lib/chain.sh lists them) on the site PRL_SITE.
# Submit through scripts/submit.sh (scripts/pbs/submit.sh), which sets the variables.
set -euo pipefail
cd "${PBS_O_WORKDIR:?submit from the repo root}"
export PRL_ROOT=$PWD PRL_SITE=${PRL_SITE:?submit through scripts/submit.sh} JOB_ID=${PBS_JOBID%%.*}
set -a; source "configs/sites/$PRL_SITE.env"; set +a
source scripts/lib/chain.sh
run_chain
