#!/bin/bash
# The quick start in one command: train Qwen3.5-9B on OpenThoughts-Agent-v1-RL with a transform, then evaluate it on
# OpenThoughts-TB-dev, on a site set up as docs/setup.md says (one node of 4 GPUs with >= 90 GB each, >= 768 GB of
# host RAM). The default transform is signed ATLAS as in the README's results (58.09% with the judge gpt-6.1-sol at
# reasoning effort low, set in the site profile). From the repo root:
#   PRL_SITE=<site> bash scripts/quickstart.sh                          # signed ATLAS
#   PRL_SITE=<site> TRANSFORM=my_transform bash scripts/quickstart.sh   # your function, transforms/my_transform.py
#   PRL_SITE=<site> TRANSFORM=atlas PRL_TRANSFORM_ARGS='{"other_weight": 0.3}' PRL_RUN_ID=atlas03 bash scripts/quickstart.sh
# One job runs the chain prepare+download+convert+smoke+train_eval (scripts/lib/chain.sh): the first time, the
# container image, Harbor and the tasks, the weights, the base checkpoint for Megatron and the framework smoke (2
# updates on 8 tasks; each skipped once done), then 68 training updates, the export and the TB-dev evaluation; about
# 15 h on 4 GB200 or H100. A job stopped by its time limit goes on where it stopped when the same command is submitted
# again (same PRL_RUN_ID). The sandboxes come first (scripts/setup/build_sandboxes.sh): with only the smoke's 8 built
# (PRL_SANDBOX_SMOKE_ONLY=1) the same command tries the transform on them instead (CHAIN=...+smoke_grpo: 2 updates).
# Variables: TRANSFORM (default atlas_signed), PRL_TRANSFORM_ARGS (JSON; atlas_signed's default
# {"failed_good_weight": 0.0}, the 58.09% run), ALGO (grpo, the default; reinforce_plus_plus; ppo after a passed
# CHAIN=smoke_ppo), PRL_RUN_ID (default quickstart), PRL_NEEDS_JUDGE and ARM (see below: when reading the transform's
# file guesses wrong); DRY_RUN=1 prints the scheduler command and submits nothing.
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
[ -f "configs/sites/$PRL_SITE.env" ] || { echo "no configs/sites/$PRL_SITE.env (copy configs/sites/example.env)" >&2; exit 1; }
profile() { (export PRL_ROOT=$PWD; set -a; source "configs/sites/$PRL_SITE.env"; printf '%s' "${!1:-}"); }
# a transform that asks a model (ask, or ATLAS' labeler) needs the judge of the profile before the job starts. Found
# by reading its file; PRL_NEEDS_JUDGE=0 / 1 says it instead (e.g. 0 for atlas with select="random", 1 for a function
# that calls ask from a module of its own)
needs_judge=${PRL_NEEDS_JUDGE:-}
if [ -z "$needs_judge" ]; then
  needs_judge=0; ! grep -qE '(^|[^a-z_])ask\(|labeler\.' $(printf 'transforms/%s.py ' ${TRANSFORM//+/ }) || needs_judge=1
fi
[[ $needs_judge == [01] ]] || { echo "PRL_NEEDS_JUDGE must be 0 or 1, got $needs_judge" >&2; exit 1; }
if [ "$needs_judge" = 1 ] &&
   { [[ "$(profile PRL_JUDGE_API_BASE)$(profile PRL_JUDGE_MODEL)" == *"<"* ]] || [ -z "$(profile PRL_JUDGE_MODEL)" ] ||
     [ ! -s "$(profile PRL_JUDGE_KEY_FILE)" ]; }; then
  echo "$TRANSFORM asks a judge: set PRL_JUDGE_API_BASE, PRL_JUDGE_MODEL and the key file PRL_JUDGE_KEY_FILE in" \
       "configs/sites/$PRL_SITE.env (or PRL_NEEDS_JUDGE=0 if it does not call one)" >&2; exit 1
fi
# a transform that rerolls (step.reroll) trains with the branch post-process of the reroll arm (GRPO or PPO); found by
# reading its file, an ARM given here wins (e.g. ARM=ota_train_transform for one whose reroll is switched off)
if [ "$ALGO" != reinforce_plus_plus ] && grep -qE '\.reroll\(' $(printf 'transforms/%s.py ' ${TRANSFORM//+/ }); then
  export ARM=${ARM:-ota_train_transform_reroll}
fi
follow() {  # RUN: the run's directory under runs/ (the arm's RUN_DIR)
  echo "[quickstart] the job's log: $(profile PRL_EXT)/logs/${CHAIN}_<job id>.log (a line per step); each update:" \
       "a line step N: {'train/loss': ...} in $(profile PRL_EXT)/runs/$1/train.log" >&2
}
# the training snapshot's templates (configs/arms/common_ota.env reads the same PRL_TRAIN_SNAPSHOT)
snapshot=$(profile PRL_TRAIN_SNAPSHOT); snapshot=${snapshot:-$(profile PRL_EXT)/task_snapshots/otagent728}
build="PRL_SITE=$PRL_SITE bash scripts/setup/build_sandboxes.sh"
[ -f "$snapshot/manifest.json" ] || { echo "no sandboxes yet ($snapshot/manifest.json): first $build" >&2; exit 1; }
ready=$(grep -c '"status": "ready"' "$snapshot/manifest.json" || true)
tasks=$(find "$snapshot/tasks" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$ready" -lt "$tasks" ]; then
  [ "$ready" -ge 8 ] || { echo "$ready of $tasks training tasks have a sandbox template: $build" >&2; exit 1; }
  [ "$ALGO" = grpo ] || { echo "ALGO=$ALGO trains on all $tasks tasks, $ready have a template: $build" >&2; exit 1; }
  echo "$ready of $tasks training tasks have a sandbox template: trying $TRANSFORM on the smoke's 8 (2 updates, about" \
       "30 min) instead of training; for the full run: $build" >&2
  if [ "$TRANSFORM" = atlas_signed ]; then  # its judge at low effort, as in the full run
    export ARM=${ARM:-ota_train_smoke_atlas_signed} PRL_TRANSFORM_ARGS=${PRL_TRANSFORM_ARGS:-'{"failed_good_weight": 0.0}'}
  fi
  # a transform that rerolls: the try's arm with the same post-process
  [ "${ARM:-}" != ota_train_transform_reroll ] || export ARM=ota_train_smoke_reroll
  # about 1 h with the image pull; 2 h, not the smoke's 8, so that it queues sooner
  export CHAIN=prepare+download+convert+smoke_grpo ELAPSE=${ELAPSE:-02:00:00}
  follow 'grpo_ota_train_smoke_<job id>'; exec bash scripts/submit.sh
fi
if [ "$TRANSFORM" = atlas_signed ]; then
  export ARM=${ARM:-ota_train_atlas_signed} PRL_TRANSFORM_ARGS=${PRL_TRANSFORM_ARGS:-'{"failed_good_weight": 0.0}'}
else
  export ARM=${ARM:-ota_train_transform}   # e.g. ARM=ota_train_transform_reroll for a transform that rerolls
fi
export CHAIN=prepare+download+convert${SMOKE_ARM:++smoke}+train_eval
follow "*_$PRL_RUN_ID"; exec bash scripts/submit.sh
