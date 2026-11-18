#!/bin/bash
# The quick start in one command: train Qwen3.5-9B on OpenThoughts-Agent-v1-RL with a transform, then evaluate it on
# OpenThoughts-TB-dev, on a site set up as docs/setup.md says (one node of 4 GPUs with >= 90 GB each, >= 768 GB of
# host RAM). The default transform is signed ATLAS as in the README's results (58.09% with the judge gpt-6.1-sol at
# reasoning effort low, set in the site profile). From the repo root:
#   PRL_SITE=<site> bash scripts/quickstart.sh                          # signed ATLAS
#   PRL_SITE=<site> TRANSFORM=my_transform bash scripts/quickstart.sh   # your function, transforms/my_transform.py
#   PRL_SITE=<site> TRANSFORM=atlas PRL_TRANSFORM_ARGS='{"other_weight": 0.3}' PRL_RUN_ID=atlas03 bash scripts/quickstart.sh
# One job runs the chain download+convert+smoke+train_eval (scripts/lib/chain.sh): the weights, the base checkpoint
# for Megatron, the framework smoke the first time (2 updates on 8 tasks; skipped once a smoke of the same code has
# passed), 68 training updates, the export and the TB-dev evaluation; about 15 h on 4 GB200 or H100. A job stopped by
# its time limit goes on where it stopped when the same command is submitted again (same PRL_RUN_ID).
# Variables: TRANSFORM (default atlas_signed), PRL_TRANSFORM_ARGS (JSON; atlas_signed's default
# {"failed_good_weight": 0.0}, the 58.09% run), ALGO (grpo, the default; reinforce_plus_plus; ppo after a passed
# CHAIN=smoke_ppo), PRL_RUN_ID (default quickstart); DRY_RUN=1 prints the scheduler command and submits nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PRL_SITE:?set PRL_SITE to a site with configs/sites/<site>.env (see configs/sites/example.env, docs/setup.md)}"
export TRANSFORM=${TRANSFORM:-atlas_signed} ALGO=${ALGO:-grpo} PRL_RUN_ID=${PRL_RUN_ID:-quickstart}
for t in ${TRANSFORM//+/ }; do [ -f "transforms/$t.py" ] || { echo "no transforms/$t.py" >&2; exit 1; }; done
# the framework smoke of this algorithm, run in the same job when none has passed yet; PPO's takes two training runs
# (CHAIN=smoke_ppo, CONTRIBUTING.md): it must have passed before
case $ALGO in
  grpo) export SMOKE_ARM=ota_train_smoke_noturncap ;;
  reinforce_plus_plus) export SMOKE_ARM=ota_train_smoke_rpp ;;
  ppo) ;;
  *) echo "ALGO must be grpo, reinforce_plus_plus or ppo, got $ALGO" >&2; exit 1 ;;
esac
if [ "$TRANSFORM" = atlas_signed ]; then
  export ARM=${ARM:-ota_train_atlas_signed} PRL_TRANSFORM_ARGS=${PRL_TRANSFORM_ARGS:-'{"failed_good_weight": 0.0}'}
else
  export ARM=${ARM:-ota_train_transform}   # e.g. ARM=ota_train_transform_reroll for a transform that rerolls
fi
export CHAIN=download+convert${SMOKE_ARM:++smoke}+train_eval
exec bash scripts/submit.sh
