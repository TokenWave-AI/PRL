#!/bin/bash
#PJM -j
# A chain job under Fujitsu PJM: the steps in PRL_CHAIN (scripts/lib/chain.sh lists them) on the site PRL_SITE.
# Submit through scripts/submit.sh (scripts/pjm/submit.sh), which sets the variables.
set -euo pipefail
cd "${PJM_O_WORKDIR:?submit from the repo root}"
export PRL_ROOT=$PWD PRL_SITE=${PRL_SITE:?submit through scripts/submit.sh} JOB_ID=${PJM_JOBID:?not a PJM job}
set -a; source "configs/sites/$PRL_SITE.env"; set +a
source scripts/lib/chain.sh
run_chain
