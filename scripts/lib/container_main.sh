#!/bin/bash
# Runs INSIDE the Miles container (see scripts/run.sh). Single node colocated (default), or multi-node:
#   NNODES>1 + RAY_HEAD_IP: this is the Ray head; the other nodes join via scripts/lib/chain_worker.sh.
#   COLOCATE=0: separate GPUs for training (ACTOR_NODES nodes x ACTOR_GPUS_PER_NODE) and rollout (ROLLOUT_GPUS SGLang engines).
#   ASYNC=1: train_async.py (rollout N+1 overlaps training N; needs COLOCATE=0 and --use-rollout-logprobs).
#   MODE=rollout  -> --debug-rollout-only: SGLang + TITO sessions + agent rollouts + grading, no training
#   MODE=train    -> a few GRPO steps (needs the torch_dist checkpoint from scripts/setup/convert_ckpt.sh)
set -euxo pipefail
: "${MODE:=rollout}" "${MODEL_SIZE:=9B}" "${NUM_GPUS:=1}" "${NUM_ROLLOUT:=1}"
: "${ATTN_BACKEND:=flash}"  # A6000/L40S (sm86/89) + head_dim 256: no TE flash/fused bwd -> use unfused
: "${ROLLOUT_BATCH:=2}" "${N_SAMPLES:=2}" "${RUN_DIR:=$PRL_EXT/runs/miles_${DATASET:-otagent}_${MODE}_$(date +%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
export RAY_TMPDIR="$PRL_SCRATCH/ray_tmp"  # /tmp/ray may belong to another user on shared nodes
# No PYTORCH_CUDA_ALLOC_CONF=expandable_segments: Miles TorchMemorySaver (colocate offload) rejects it.
export PYTHONUNBUFFERED=1 no_proxy=127.0.0.1,localhost CUDA_DEVICE_MAX_CONNECTIONS=1
# Ray kills workers at 95% node memory by default; colocated 4B on a GH200 node peaks near that (host offload).
# On GH200 the GPU's HBM is exposed as a system NUMA node, so Ray's monitor counts GPU allocations as host
# memory and OOM-kills the trainer at the first colocated train step (process RSS was only ~80 GB). Disabled
# by default (RAY_MEM_MONITOR_MS=0); the kernel OOM killer still guards the real host memory.
# glibc per-thread arenas bloat RSS in the heavily threaded rollout process (100+ concurrent episodes)
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export RAY_memory_usage_threshold=${RAY_MEM_THRESHOLD:-0.98} RAY_memory_monitor_refresh_ms=${RAY_MEM_MONITOR_MS:-0}
# Ray async actors run on uvloop by default; the RolloutExecutor (hundreds of Harbor trials, aiohttp + threads)
# segfaulted in uvloop's TimerHandle.cancel (libuv heap_remove) after 20 steps of a run. Plain asyncio instead.
export RAY_USE_UVLOOP=0
# A process killed with SIGABRT (or crashing) prints every thread's Python stack (the stall guard, scripts/lib/guard.sh)
export PYTHONFAULTHANDLER=1

# Model profile (configs/models/$MODEL.env): HF dir, Megatron args, ref checkpoint, TITO model, SGLang parsers.
# run.sh passes MODEL and MODEL_HF_DIR as resolved on the host; a launch that does not gets the profile of
# MODEL_SIZE, as before MODEL existed.
: "${MODEL:=qwen3.5-${MODEL_SIZE}}"
set -a; source "$PRL_ROOT/configs/models/$MODEL.env"; set +a
HF=$MODEL_HF_DIR
[ -f "$HF/config.json" ] || { echo "MODEL=$MODEL: no HF checkpoint at '$HF' (download $MODEL_HF_REPO)" >&2; exit 1; }
# Past max_position_embeddings the positions are untrained (no rope scaling is configured on either side).
[ "${MAX_CONTEXT:-32768}" -le "$MODEL_MAX_CONTEXT" ] || { echo "MAX_CONTEXT=$MAX_CONTEXT > $MODEL_MAX_CONTEXT ($MODEL)" >&2; exit 1; }
cd "${PRL_MILES_DIR:-$PRL_ROOT/build/miles}"   # patched copy (scripts/setup/apply_patches.sh), not the pristine vendored tree
read -ra MODEL_ARGS <<< "$(python3 miles/utils/external_utils/model_args_utils.py "$MODEL_ARGS_NAME")"

CKPT_ARGS=(--hf-checkpoint "$HF")
if [ "$MODE" = "train" ]; then
  CKPT_ARGS+=(--ref-load "$MODEL_TORCH_DIST" --load "$RUN_DIR/ckpt" --save "$RUN_DIR/ckpt")
  [ -z "${SAVE_INTERVAL:-}" ] && CKPT_ARGS+=(--save-interval 100000)
fi

ROLLOUT_ARGS=(
  --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata
  --rollout-shuffle --rollout-seed "${ROLLOUT_SEED:-42}" --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH" --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len "${MAX_RESPONSE:-$([ "${THINKING:-false}" = true ] && echo 8192 || echo 4096)}" --max-seq-len "${MAX_CONTEXT:-32768}"
  --rollout-temperature "${TEMPERATURE:-1.0}" --rollout-top-p "${TOP_P:-1.0}"
  --global-batch-size $((ROLLOUT_BATCH * N_SAMPLES))
  --dump-details "$RUN_DIR/traces"
)
# Harbor trial in the rollout worker (Terminus-2 by default), sandboxes over the E2B API (prebuilt
# per-task templates); reward = the task verifier. See prl/rollout/harbor_agent.py.
[ "${HARNESS:-}" = harbor ] || { echo "HARNESS=${HARNESS:-} : only harbor is supported (configs/datasets/otagent.env)" >&2; exit 1; }
AGENT_ARGS=(
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
  --custom-agent-function-path prl.rollout.harbor_agent.run
  --custom-rm-path generate.reward_func
  --dynamic-sampling-filter-path "$([ "${DYN_FILTER:-none}" = std ] && echo miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter || echo miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted)"
  --use-session-server --session-server-port 30000 --session-server-workers "${SESSION_WORKERS:-4}"
  --tito-model "$TITO_MODEL"
  --apply-chat-template-kwargs "{\"enable_thinking\": ${THINKING:-false}}"
)
# The Harbor recipe's generate.RolloutFn (agent metrics), refilling a dropped group with one prompt instead of a new
# batch (prl/rollout/refill.py); --fully-async brings its own rollout function and refuses another
[[ " ${EXTRA_ARGS:-} " = *" --fully-async "* ]] || AGENT_ARGS+=(--rollout-function-path prl.rollout.refill.RolloutFn)
GRPO_ARGS=(
  --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}"
  --custom-advantage-postprocess-path prl.algo.step_advantage.step_weight_postprocess
  --kl-loss-coef "${KL_LOSS_COEF:-0.0}" --kl-coef 0.0 --entropy-coef 0.0 --eps-clip "${EPS_CLIP:-0.2}"
)
# PPO (ADVANTAGE_ESTIMATOR=ppo): Miles builds a critic on the actor's GPUs (examples/ppo/README.md). Value pretraining
# = CRITIC_WARMUP_STEPS rollouts that train only the critic on the frozen actor's own samples.
if [ "${ADVANTAGE_ESTIMATOR:-grpo}" = ppo ] && [ "$MODE" = train ]; then
  source "$PRL_ROOT/scripts/lib/relay.sh"
  CRITIC_LOAD=$(critic_load "$RUN_DIR" "$MODEL_TORCH_DIST")
  GRPO_ARGS+=(--critic-load "$CRITIC_LOAD" --critic-save "$RUN_DIR/ckpt_critic" --critic-lr "${CRITIC_LR:?}"
              --num-critic-only-steps "${CRITIC_WARMUP_STEPS:?}" --gamma "${GAE_GAMMA:?}" --lambd "${GAE_LAMBDA:?}"
              --value-clip "${VALUE_CLIP:-0.2}")
  [ "${NORMALIZE_ADVANTAGES:-0}" = 1 ] && GRPO_ARGS+=(--normalize-advantages)
fi
# REINFORCE++ (ota_train_full_rpp_b32): Miles refuses the estimator without whitening over the batch
[ "${ADVANTAGE_ESTIMATOR:-grpo}" = reinforce_plus_plus ] && [ "${NORMALIZE_ADVANTAGES:-0}" = 1 ] && GRPO_ARGS+=(--normalize-advantages)
# KL loss against the reference model (OpenThinker/SkyRL: use_kl_loss, k3, coef 0.001)
[ "${KL_LOSS_COEF:-0}" != 0 ] && [ "${KL_LOSS_COEF:-0}" != 0.0 ] && GRPO_ARGS+=(--use-kl-loss --kl-loss-type "${KL_LOSS_TYPE:-k3}")
# train/entropy_loss: the policy entropy over the loss-masked tokens, logged only (entropy coef 0); default on (scripts/lib/env.sh)
[ "${PRL_OBSERVE_ENTROPY:-0}" = 1 ] && GRPO_ARGS+=(--observe-training-entropy)
# Reward std/min/max, prompt length, per-trial output tokens and turns, sampled-token NLL, ATLAS kept shares per update
GRPO_ARGS+=(--custom-rollout-log-function-path prl.rollout.rollout_metrics.log_rollout_data)
OPT_ARGS=(--optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 "${ADAM_BETA2:-0.98}")
PERF_ARGS=(
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size "${CP_SIZE:-1}"
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
)
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${SGLANG_MEM:-0.5}"
  --sglang-context-length "${MAX_CONTEXT:-32768}"
  # Must match --tito-model (e.g. Qwen35TITOTokenizer.tool_call_parser); not auto-applied in production.
  --sglang-tool-call-parser "$SGLANG_TOOL_CALL_PARSER" --sglang-reasoning-parser "$SGLANG_REASONING_PARSER"
)
# An sgl_kernel without FA3 (flash_ops; seen on arm64 builds): SGLANG_ATTN=flashinfer in the site profile
[ -n "${SGLANG_ATTN:-}" ] && SGLANG_ARGS+=(--sglang-attention-backend "$SGLANG_ATTN")
# ...and its vision encoder (captured in SGLang's prefill CUDA-graph warmup) defaults to FA3 too: SGLANG_MM_ATTN=sdpa
[ -n "${SGLANG_MM_ATTN:-}" ] && SGLANG_ARGS+=(--sglang-mm-attention-backend "$SGLANG_MM_ATTN")
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend "$ATTN_BACKEND")
EXTRA=()
[ "$MODE" = "rollout" ] && EXTRA+=(--debug-rollout-only)
[ "${SKIP_EVAL_BEFORE_TRAIN:-0}" = 1 ] && EXTRA+=(--skip-eval-before-train)
[ -n "${OVER_SAMPLING_BATCH:-}" ] && EXTRA+=(--over-sampling-batch-size "$OVER_SAMPLING_BATCH")
if [ -n "${EVAL_DATA:-}" ]; then   # held-out eval inside training (same harness; Tmax eval sampling by default)
  EXTRA+=(--eval-prompt-data heldout "$EVAL_DATA" --eval-interval "${EVAL_INTERVAL:-20}"
          --n-samples-per-eval-prompt "${EVAL_N_SAMPLES:-1}" --eval-temperature "${EVAL_TEMPERATURE:-0.7}" --eval-top-p "${EVAL_TOP_P:-0.95}")
fi
[ -n "${SAVE_INTERVAL:-}" ] && [ "$MODE" = "train" ] && CKPT_ARGS+=(--save-interval "$SAVE_INTERVAL")
[ "${USE_TB:-1}" = 1 ] && export TENSORBOARD_DIR="$RUN_DIR/tb" && EXTRA+=(--use-tensorboard --tb-project-name prl --tb-experiment-name run)
# Free-form extra Miles flags (e.g. "--disable-grpo-std-normalization --calculate-per-token-loss")
[ -n "${EXTRA_ARGS:-}" ] && read -ra _XA <<< "$EXTRA_ARGS" && EXTRA+=("${_XA[@]}")

: "${NNODES:=1}" "${COLOCATE:=1}" "${ASYNC:=0}"
if [ "$COLOCATE" = 1 ]; then
  # NUM_GPUS per node; on NNODES nodes (e.g. the MoE arm ota_train_smoke_moe) training and rollout share them all
  PLACEMENT=(--actor-num-nodes "$NNODES" --actor-num-gpus-per-node "$NUM_GPUS" --rollout-num-gpus "$((NNODES * NUM_GPUS))" --colocate)
else
  # ACTOR_GPUS_PER_NODE: 1 on one-GPU nodes (e.g. one GH200 per node); a 4-GPU node splits its GPUs between training
  # and rollout.
  : "${ACTOR_NODES:=1}" "${ACTOR_GPUS_PER_NODE:=1}" "${ROLLOUT_GPUS:=$((NNODES - ACTOR_NODES))}"
  PLACEMENT=(--actor-num-nodes "$ACTOR_NODES" --actor-num-gpus-per-node "$ACTOR_GPUS_PER_NODE" --rollout-num-gpus "$ROLLOUT_GPUS")
fi
TRAIN_PY=train.py; [ "$ASYNC" = 1 ] && TRAIN_PY=train_async.py

# ~36 extra python3 processes (~16 GB together) appear on every 9B trainer node at the first train step, most likely
# torch inductor's compile worker pool; the trainer cannot spare that (100 GB per node), so compile in-process.
# ps_<host>.log (node_worker.sh) shows what they are
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
# The packages this job ran with (scripts/run_job.sh keeps the rest of the job's provenance there)
[ -n "${PRL_PROVENANCE_DIR:-}" ] && { python3 -m pip freeze > "$PRL_PROVENANCE_DIR/pip_freeze.txt" 2>&1 || true; }
# NCCL flight recorder: on a collective timeout every rank dumps its recent collectives to RUN_DIR/nccl_trace/ (a hang
# in the trainer leaves evidence; the rollout side has stack_dumps/, prl/rollout/harbor_fixes.py)
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000} TORCH_NCCL_DUMP_ON_TIMEOUT=1
[ -n "${RUN_DIR:-}" ] && mkdir -p "$RUN_DIR/nccl_trace" && export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$RUN_DIR/nccl_trace/rank_
ray stop --force >/dev/null 2>&1 || true
# PRL_RAY_PLASMA_DIR: the object store in a directory instead of /dev/shm. Under some PJM jobs' limits the raylet
# computes /dev/shm's free space as INT64_MIN and aborts (plasma_allocator.cc:76).
RAY_STORE=()
if [ -n "${PRL_RAY_PLASMA_DIR:-}" ]; then
  mkdir -p "$PRL_RAY_PLASMA_DIR"
  export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
  RAY_STORE=(--plasma-directory="$PRL_RAY_PLASMA_DIR" --object-store-memory="${PRL_RAY_OBJECT_STORE:-100000000000}")
fi
# Multi-node on a site naming its inter-node interface (PRL_NODE_IFACE, e.g. ib0; configs/sites/example.env): NCCL
# and gloo use it for their sockets too. Unpinned, NCCL usually picks ib0 itself, but nodes may also have bridges with
# the same 192.168.x address on every node (virbr0, krm-virbr0), which would break the bootstrap if picked (on 2 nodes,
# NCCL on ib0 ran over InfiniBand, mlx5_0/1, ~45 GB/s). The worker nodes do the same (scripts/lib/chain_worker.sh).
if [ "$NNODES" -gt 1 ] && [ -n "${PRL_NODE_IFACE:-}" ]; then
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$PRL_NODE_IFACE} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$PRL_NODE_IFACE}
fi
ray start --head --node-ip-address "${RAY_HEAD_IP:-127.0.0.1}" --port 6379 --num-gpus "$NUM_GPUS" --disable-usage-stats \
  --include-dashboard=false --temp-dir "$PRL_SCRATCH/ray_tmp" "${RAY_STORE[@]}"
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
# Point the driver at this cluster: with a custom --temp-dir / non-loopback head IP, ray.init() does not discover it
# and silently starts a private local instance (1 GPU), so a multi-GPU placement group waits forever.
export RAY_ADDRESS="${RAY_HEAD_IP:-127.0.0.1}:6379"
if [ "$NNODES" -gt 1 ]; then   # wait for the worker nodes (scripts/lib/chain_worker.sh) to join
  for i in $(seq 120); do
    n=$(python3 -c "import os, ray; ray.init(address=os.environ.get(\"RAY_ADDRESS\",\"auto\"), logging_level=\"ERROR\"); print(sum(1 for x in ray.nodes() if x['Alive']))" 2>/dev/null || echo 0)
    [ "$n" -ge "$NNODES" ] && break; sleep 5
  done
  echo "ray nodes alive: $n / $NNODES"; [ "$n" -ge "$NNODES" ] || { echo "worker nodes did not join"; exit 1; }
fi

python3 "$TRAIN_PY" \
  "${PLACEMENT[@]}" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${AGENT_ARGS[@]}" "${GRPO_ARGS[@]}" "${OPT_ARGS[@]}" \
  "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
