#!/bin/bash
# Chain step after a smoke (scripts/lib/chain.sh): CPU acceptance of the smoke run (scripts/checks/
# grpo_acceptance.py), whose receipt (acceptance.json) vouches for these sources (scripts/checks/find_smoke.py).
# PRL_SMOKE_JOB=<job>; a smoke arm with another RUN_DIR (e.g. ota_train_smoke_moe) also
# PRL_SMOKE_RUN=<its RUN_DIR>.
set -euo pipefail
export PRL_ROOT=${PRL_ROOT:-$PWD} PRL_SITE=${PRL_SITE:?set PRL_SITE (configs/sites)}
source "$PRL_ROOT/configs/sites/$PRL_SITE.env"
: "${PRL_SMOKE_JOB:?Set the smoke job ID}"
export PRL_SMOKE_RUN=${PRL_SMOKE_RUN:-$PRL_EXT/runs/grpo_ota_train_smoke_${PRL_SMOKE_JOB}}
exec "$PRL_HOST_PYTHON" "$PRL_ROOT/scripts/checks/grpo_acceptance.py"
