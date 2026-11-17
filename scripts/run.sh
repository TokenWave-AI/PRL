#!/bin/bash
# Host-side launcher: runs scripts/lib/container_main.sh in the Miles container (called by scripts/run_job.sh, which
# sources the site, the arm and scripts/lib/env.sh first). Everything here is a hyperparameter forwarded into the
# container.
set -euo pipefail
source "$(dirname "$0")/lib/env.sh"
mkdir -p "$PRL_SCRATCH/ray_tmp"
# Knobs forwarded into the container when set (see scripts/lib/container_main.sh and README).
PASS=()
for v in RUN_DIR NUM_GPUS ROLLOUT_SEED TEMPERATURE TOP_P MAX_RESPONSE LR EPS_CLIP DYN_FILTER OVER_SAMPLING_BATCH \
         EVAL_DATA EVAL_INTERVAL EVAL_N_SAMPLES EVAL_TEMPERATURE EVAL_TOP_P SAVE_INTERVAL USE_TB EXTRA_ARGS \
         SGLANG_ATTN SGLANG_MM_ATTN RAY_MEM_THRESHOLD RAY_MEM_MONITOR_MS NNODES COLOCATE ASYNC ACTOR_NODES ROLLOUT_GPUS CP_SIZE \
         RAY_HEAD_IP PRL_NODE_IFACE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME; do
  [ -n "${!v:-}" ] && PASS+=("$v=${!v}")
done
# Host-side settings (paths, site and launcher options) share the PRL_ prefix but stay on the host; the container
# gets its paths from the explicit list below
HOST_ONLY='PRL_(ROOT|EXT|SCRATCH|SITE|SCHEDULER|SIF|CHAIN|MILES_DIR|MILES_EXTRA_ARGS|EXTRA_BINDS|EXTRA_PYTHONPATH|PYDEPS|SKIP_APPLY_PATCHES|SANDBOX_UNSQUASH|SUBMIT_FINDS_SMOKE|REMOTE_EXEC|NODE_FILE|ARCHIVE_TRIALS|WORKER_JOIN_S|PJM_[A-Z_]+|PBS_[A-Z_]+|SINGULARITY)'
# Harbor harness (HARNESS=harbor): the trial, sandbox (E2B API) and litellm knobs; every PRL_* knob (the
# container runs with --cleanenv: one not forwarded silently keeps its default, as PRL_LENGTH_LAMBDA did,
# tests/unit/test_run_passthrough.py); PPO (PRL_VALUE_HEAD_INIT: patches miles/0006, 0008)
for v in ACTOR_GPUS_PER_NODE HARNESS KL_LOSS_COEF KL_LOSS_TYPE ADAM_BETA2 SKIP_EVAL_BEFORE_TRAIN SESSION_WORKERS \
         ADVANTAGE_ESTIMATOR CRITIC_LR CRITIC_WARMUP_STEPS GAE_GAMMA GAE_LAMBDA VALUE_CLIP NORMALIZE_ADVANTAGES PRL_VALUE_HEAD_INIT \
         $(compgen -e | grep -E '^(HARBOR_|E2B_|AGENT_|LITELLM_|PRL_)' | grep -vxE "$HOST_ONLY"); do
  [ -n "${!v:-}" ] && PASS+=("$v=${!v}")
done
in_miles env MODE="${MODE:-rollout}" MODEL_SIZE="${MODEL_SIZE:-9B}" NUM_GPUS="${NUM_GPUS:-1}" \
  NUM_ROLLOUT="${NUM_ROLLOUT:-1}" ROLLOUT_BATCH="${ROLLOUT_BATCH:-2}" N_SAMPLES="${N_SAMPLES:-2}" \
  ATTN_BACKEND="${ATTN_BACKEND:-flash}" MAX_CONTEXT="${MAX_CONTEXT:-32768}" MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}" \
  SGLANG_MEM="${SGLANG_MEM:-0.5}" THINKING="${THINKING:-false}" \
  PRL_ROOT="$PRL_ROOT" PRL_EXT="$PRL_EXT" PRL_SCRATCH="$PRL_SCRATCH" DATASET="$DATASET" PROMPT_DATA="$PROMPT_DATA" \
  QWEN35_9B="${QWEN35_9B:-}" MODEL="$MODEL" MODEL_HF_DIR="$MODEL_HF_DIR" \
  PRL_REJECT_MISMATCH="${PRL_REJECT_MISMATCH:-0}" PRL_MISMATCH_MAX_RETRIES="${PRL_MISMATCH_MAX_RETRIES:-2}" \
  PRL_ABORT_MAX_RETRIES="${PRL_ABORT_MAX_RETRIES:-2}" \
  "${PASS[@]}" \
  bash "$(cd "$(dirname "$0")" && pwd)/lib/container_main.sh"
