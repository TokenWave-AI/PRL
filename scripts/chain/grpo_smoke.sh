#!/bin/bash
# Chain step smoke / smoke_grpo / smoke_ppo (scripts/lib/chain.sh): the framework smoke on this node. Compute-node gates
# (hardware, real Miles sample contracts, validated CPU oracle/nop receipt), then 2 synchronous GRPO updates (ARM, default
# ota_train_smoke; e.g. ota_train_smoke_transform with TRANSFORM=<name>, ota_train_smoke_moe with
# MODEL=qwen3.6-35B-A3B). Aborts before training on any failed gate. scripts/chain/grpo_acceptance.sh then checks the run.
set -euo pipefail
export PRL_ROOT=${PRL_ROOT:-$PWD} PRL_SITE=${PRL_SITE:?set PRL_SITE (configs/sites)}
set -a
source "$PRL_ROOT/configs/sites/$PRL_SITE.env"
source "$PRL_ROOT/configs/arms/common_ota_single_node.env"
ARM=${ARM:-ota_train_smoke}
source "$PRL_ROOT/configs/arms/$ARM.env"
set +a
source "$PRL_ROOT/scripts/lib/env.sh"   # model profile (MODEL_*) for the preflight; run_job.sh sources it again
# Resume half of the PPO smoke (ota_train_smoke_ppo): a dependent job may start whatever the first job's exit
# status, so run only on that job's actor and critic checkpoints.
if [ -n "${PRL_SMOKE_RESUME_FROM:-}" ]; then
  source "$PRL_ROOT/scripts/lib/relay.sh"
  for sub in ckpt ckpt_critic; do
    [ "$(latest_iter "$RUN_DIR" "$sub")" = "$PRL_SMOKE_RESUME_FROM" ] ||
      { echo "[smoke] $RUN_DIR/$sub is not at iteration $PRL_SMOKE_RESUME_FROM" >&2; exit 1; }
  done
fi
if ! command -v apptainer >/dev/null 2>&1; then module load apptainer; fi
mkdir -p "$RUN_DIR" "$PRL_SCRATCH"
source "$PRL_ROOT/scripts/lib/guard.sh"
hardware_gate
CONTAINER_PYTHONPATH="$PRL_ROOT:$PRL_ROOT/build/miles:$PRL_ROOT/build/miles/examples/experimental/harbor:$PRL_EXT/pydeps_miles:$PRL_EXT/pydeps_e2b_sdk:$PRL_EXT/pydeps_harbor:/root/Megatron-LM"
# as the training container (scripts/lib/env.sh in_miles): the repo, PRL_EXT and PRL_EXTRA_BINDS (e.g. where the key
# files are)
binds=(-B "$PRL_ROOT" -B "$PRL_EXT"); for d in ${PRL_EXTRA_BINDS:-}; do if [ -e "$d" ]; then binds+=(-B "$d"); fi; done
apptainer exec --nv "${binds[@]}" \
  --env "PYTHONPATH=$CONTAINER_PYTHONPATH" \
  "$MILES_IMAGE" /opt/sglang/bin/python3 "$PRL_ROOT/scripts/checks/grpo_preflight.py"
# A trajectory transform (ARM=ota_train_smoke_transform, TRANSFORM=<name>) runs on recorded trials first,
# as every training job does (scripts/lib/transform_check.sh)
source "$PRL_ROOT/scripts/lib/transform_check.sh"
transform_check
run_guarded smoke
