# A chain of steps in one batch job, shared by the scheduler wrappers scripts/pjm/job.sh (Fujitsu PJM),
# scripts/pbs/job.sh (PBS Pro) and scripts/local/job.sh (this machine, no scheduler). The chain steps
# (scripts/chain/*.sh) do the work with PRL_SITE set; a one-node job may run 168 h, so a chain needs neither the 24 h
# relay nor dependent jobs. The wrapper
# sets PRL_ROOT, PRL_SITE, JOB_ID and sources configs/sites/$PRL_SITE.env, then calls run_chain. Submit through the
# site's submit.sh (scripts/lib/submit_chain.sh builds the variables).
# PRL_CHAIN is one step or several joined by "+" (e.g. download+convert+smoke+train_eval), run in order:
#   prepare      once per site: the Miles image when MILES_IMAGE is missing (PRL_MILES_IMAGE_SOURCE, default
#                docker://radixark/miles:v0.1.2), Harbor and its dependencies for the container
#                (scripts/setup/install_harbor_deps.sh) and the tasks (scripts/setup/prepare_otagent.sh); each skips
#                what is there
#   download     MODEL's weights into $HF_HOME/hub (scripts/setup/download_model.sh; hf skips what is there)
#   convert      once per model: container check, then the torch_dist base checkpoint; skipped when it exists
#   smoke        plain framework smoke for this job's model and context (SMOKE_ARM, e.g. ota_train_smoke_moe)
#                + its acceptance; skipped when an accepted smoke already vouches for these sources, MODEL and ARM's
#                context (scripts/checks/find_smoke.py). Both smokes first check their 8 tasks' templates (oracle /
#                no-op) when the snapshot has no such result yet (smoke_contract)
#   smoke_grpo   framework smoke of a transform (ARM=ota_train_smoke_transform, TRANSFORM=...) + acceptance
#   smoke_ppo    the same with ALGO=ppo: job A (2 rollouts), job B (resume, 3 rollouts), acceptance
#   train_eval   grpo_full.sh on RUN_DIR (resumes what is there); once the last iteration is saved, the
#                export (ckpt_to_hf.sh) and the TB-dev 70x8 evaluation (tbdev70_eval.sh with EVAL_ARM, default
#                ota_eval_tbdev70_64k_summary_c16_noturncap)
#   eval_base    TB-dev 70x8 of MODEL's base weights with EVAL_ARM: the baseline of train_eval's evaluation
#   eval_hf      TB-dev 70x8 of an exported checkpoint (MODEL_HF_DIR, from train_eval's export) with EVAL_ARM: a
#                re-evaluation without the training (e.g. when the training job's own evaluation lost trials)
#   task_check   oracle / nop Harbor trials on prebuilt task templates (PRL_TEMPLATE_MANIFEST, PRL_TASK_ROOT;
#                scripts/chain/task_smoke.sh): before a new template snapshot is used. Without them: the framework
#                smoke's 8 tasks into <training snapshot>/contract, as the smokes do first (smoke_contract)
#   unit         the unit tests in the Miles image
#   replay       train-only checks and failure drills on an accepted smoke's batches (PRL_REPLAY_SOURCE, its
#                RUN_DIR; scripts/chain/replay_check.sh): resume, data parallelism, longest sample, NaN, kills
# download and convert hold a lock per model, so jobs of one model that start together do it once.
# When the chain ends (also after a failed step or the scheduler's SIGTERM), this job's own trial directories (the
# training arm's HARBOR_TRIALS_DIR, the evaluation's) are packed into tars (scripts/lib/archive_trials.sh;
# PRL_ARCHIVE_TRIALS=0 keeps them as directories).
source scripts/lib/archive_trials.sh

# in_arm ARM SCRIPT [ARG...]: run the bash SCRIPT ($2... are the ARGs) with ARM resolved as the chain steps resolve
# it (site, common, arm), from a clean environment that has only the variables naming a run
in_arm() {
  local arm=$1 script=$2; shift 2
  env -i PATH=/usr/bin:/bin HOME="$HOME" PRL_ROOT="$PRL_ROOT" PRL_SITE="$PRL_SITE" MODEL="${MODEL:?}" TRANSFORM="${TRANSFORM:-}" \
    ALGO="${ALGO:-grpo}" PRL_RUN_ID="${PRL_RUN_ID:-}" JOB_ID="${JOB_ID:-0}" PJM_SSD_DIR="${PJM_SSD_DIR:-}" \
    ${PRL_EXT:+PRL_EXT="$PRL_EXT"} ${PRL_SCRATCH:+PRL_SCRATCH="$PRL_SCRATCH"} bash -c 'set -a
    source "configs/sites/$PRL_SITE.env"; source configs/arms/common_ota_single_node.env; source "configs/arms/$0.env"
    set +a; eval "$1"' "$arm" "$script" "$@"
}

# arm_setting VAR [ARM]: a setting of an arm (default $ARM)
arm_setting() {
  in_arm "${2:-${ARM:?}}" 'echo "${!2}"' "$1"
}

# find_smoke [ARM]: the RUN_DIR of the newest accepted smoke for this algorithm, MODEL and the arm's context and
# schedule (ASYNC; default $ARM), or nothing (scripts/checks/find_smoke.py)
find_smoke() {
  ALGO=${ALGO:-grpo} MODEL=$MODEL MAX_CONTEXT=$(arm_setting MAX_CONTEXT "${1:-${ARM:?}}") ASYNC=$(arm_setting ASYNC "${1:-${ARM:?}}") \
    "$PRL_HOST_PYTHON" scripts/checks/find_smoke.py
}

# sandbox: SingularityCE runs the image as us, and its /root (Megatron-LM) is root-only (drwx------): unpack the image
# ($MILES_IMAGE) on this node's NVMe, owned by us; scripts/lib/env.sh and the launchers then use the directory
sandbox() {
  local sif=${PRL_SIF:=$MILES_IMAGE}
  local sb=$PRL_SCRATCH/sandbox_$(basename "$sif" .sif)
  if [ ! -d "$sb" ]; then
    mkdir -p "$PRL_SCRATCH/stmp"
    if [ "${PRL_SANDBOX_UNSQUASH:-0}" = 1 ]; then
      # Where build --sandbox stops at the pulled image's /.singularity.d/runscript symlink ("file exists", seen with
      # SingularityCE on aarch64 nodes): unpack its squashfs partition directly (files become ours, so /root is
      # readable)
      local offset; offset=$(apptainer sif list "$sif" | awk -F'|' '/Squashfs/ {split($4, p, "-"); print p[1] + 0}')
      unsquashfs -q -n -no-xattrs -o "${offset:?no squashfs partition in $sif}" -d "$sb" "$sif" \
        > "$PRL_SCRATCH/sandbox_build.log" 2>&1 || { tail -20 "$PRL_SCRATCH/sandbox_build.log"; rm -rf "$sb"; return 1; }
    else
      # both names: the site's apptainer wrapper copies APPTAINER_TMPDIR (configs/sites, not yet created) over the other
      APPTAINER_TMPDIR=$PRL_SCRATCH/stmp SINGULARITY_TMPDIR=$PRL_SCRATCH/stmp apptainer build --sandbox "$sb" "$sif" \
        > "$PRL_SCRATCH/sandbox_build.log" 2>&1 || { tail -20 "$PRL_SCRATCH/sandbox_build.log"; return 1; }
    fi
  fi
  export MILES_IMAGE=$sb
  echo "[$PRL_SITE] image unpacked to $sb at $(date)"
}

# start_workers ARM: an arm on several nodes (NNODES > 1; e.g. ota_train_smoke_moe) gets this job's other
# nodes as Ray workers (scripts/lib/chain_worker.sh, started with the site's $PRL_REMOTE_EXEC): each unpacks the image
# first (waited for, so the head's wait for its workers, scripts/lib/container_main.sh, covers only their start), then
# joins the cluster the launcher starts here; RAY_HEAD_IP is this node's address on $PRL_NODE_IFACE. The workers get the
# arm as resolved here ($RUN_DIR/worker.env, mode 600), without the Harbor, sandbox and model-API variables: the
# rollout manager and its trials stay on this node (--pin-rollout-manager-to-head). One node: nothing to do.
start_workers() {
  local arm=$1 nnodes run n; nnodes=$(arm_setting NNODES "$arm")
  [ "${nnodes:-1}" -gt 1 ] || return 0
  : "${PRL_REMOTE_EXEC:?$PRL_SITE cannot start a multi-node job}" "${PRL_NODE_FILE:?no node list}" "${PRL_NODE_IFACE:?}"
  local nodes=($(sort -u "$PRL_NODE_FILE")) head mine
  [ "${#nodes[@]}" = "$nnodes" ] || { echo "$arm wants $nnodes nodes, this job has ${#nodes[@]}: ${nodes[*]}" >&2; return 1; }
  # The head is the entry naming this node: a PJM node file may list addresses (e.g. of eno1), not host names, so
  # match this node's own addresses as well as its name
  mine=$( { hostname -s; /usr/sbin/ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1; } )
  head=$(printf '%s\n' "${nodes[@]}" | grep -Fxf <(echo "$mine") | head -1 || true)
  [ -n "$head" ] || { echo "this node ($(hostname -s)) is not in ${nodes[*]}" >&2; return 1; }
  run=$(arm_setting RUN_DIR "$arm"); mkdir -p "$run"
  RAY_HEAD_IP=$(/usr/sbin/ip -4 -o addr show "$PRL_NODE_IFACE" | awk '{print $4}' | cut -d/ -f1)
  [ -n "$RAY_HEAD_IP" ] || { echo "no address on $PRL_NODE_IFACE" >&2; return 1; }
  export RAY_HEAD_IP
  ( umask 077
    in_arm "$arm" 'for v in $(compgen -e | grep -E "^(PRL_|MODEL|HF_HOME$|NUM_GPUS$|NNODES$|LD_LIBRARY_PATH$|DATASET$|HARNESS$|APPTAINER_)" |
                   grep -vE "KEY|SECRET|TOKEN$|PASSWORD|^PRL_SIF$"); do declare -p "$v"; done' > "$run/worker.env" )
  local pids=() rc=0
  for n in "${nodes[@]}"; do
    [ "$n" = "$head" ] && continue
    $PRL_REMOTE_EXEC "$n" bash "$PRL_ROOT/scripts/lib/chain_worker.sh" prepare "$run" & pids+=($!)
  done
  for n in "${pids[@]}"; do wait "$n" || rc=1; done
  [ "$rc" = 0 ] || { echo "a worker node could not prepare (see $run/worker_*.log)" >&2; return 1; }
  for n in "${nodes[@]}"; do
    [ "$n" = "$head" ] && continue
    $PRL_REMOTE_EXEC "$n" bash "$PRL_ROOT/scripts/lib/chain_worker.sh" join "$run" "$RAY_HEAD_IP" > /dev/null 2>&1 &
  done
  echo "[$PRL_SITE] workers ${nodes[*]/$head} joining $RAY_HEAD_IP (logs $run/worker_*.log)"
}

T=scripts/chain
EVAL_ARM=${EVAL_ARM:-ota_eval_tbdev70_64k_summary_c16_noturncap}

# smoke_contract ARM: the smoke's preflight needs the oracle / no-op results of its 8 tasks on their templates
# (<snapshot>/contract, scripts/chain/task_smoke.sh); a snapshot without them gets them first. They are written to a
# temporary directory and moved into place only when every task passed, so a failed check is rerun next time.
smoke_contract() {
  local manifest contract
  manifest=$(arm_setting PRL_TEMPLATE_MANIFEST "$1"); contract=$(dirname "$manifest")/contract
  if [ -d "$contract" ]; then echo "[$PRL_SITE] $contract: the smoke's tasks are checked already"; return 0; fi
  echo "[$PRL_SITE] $contract missing: oracle / no-op trials of the smoke's tasks first"
  rm -rf "$contract.partial"
  PRL_TEMPLATE_MANIFEST=$manifest PRL_TASK_ROOT=$(arm_setting HARBOR_TASKS_DIR "$1") \
    PRL_TASK_PROMPTS=$(arm_setting PROMPT_DATA "$1") PRL_TASK_CHECK_OUTPUT=$contract.partial bash $T/task_smoke.sh
  mv "$contract.partial" "$contract"
}
run_step() {
  case $1 in
  prepare)  # once per site, on a node with internet: the Miles image when MILES_IMAGE is missing (mksquashfs needs
    # ~74 GB of RAM, so not on a login node), Harbor, the e2b SDK and litellm for the container (skipped when
    # installed), then the tasks as Harbor dirs and prompt rows (prepare_otagent.sh skips what is there)
    if [ ! -e "$MILES_IMAGE" ]; then
      local partial=${MILES_IMAGE%.sif}.partial.sif
      mkdir -p "$(dirname "$MILES_IMAGE")" "$PRL_SCRATCH/pull_tmp" "$PRL_SCRATCH/pull_cache"; rm -f "$partial"
      echo "[$PRL_SITE] pulling ${PRL_MILES_IMAGE_SOURCE:=docker://radixark/miles:v0.1.2} to $MILES_IMAGE"
      APPTAINER_TMPDIR=$PRL_SCRATCH/pull_tmp APPTAINER_CACHEDIR=$PRL_SCRATCH/pull_cache \
        apptainer pull "$partial" "$PRL_MILES_IMAGE_SOURCE"
      mv "$partial" "$MILES_IMAGE"; rm -rf "$PRL_SCRATCH/pull_tmp" "$PRL_SCRATCH/pull_cache"
    fi
    sandbox
    bash scripts/setup/install_harbor_deps.sh
    MODELS= bash scripts/setup/prepare_otagent.sh ;;
  download)
    sandbox
    ( exec 8>"$PRL_EXT/locks/model_$MODEL.lock"; flock 8
      DATASET=otagent bash scripts/setup/download_model.sh | tail -3 ) ;;
  convert)  # the container with in_miles' flags, then HF -> torch_dist (scripts/setup/convert_ckpt.sh)
    sandbox
    ( exec 8>"$PRL_EXT/locks/model_$MODEL.lock"; flock 8
      export DATASET=${DATASET:-otagent}   # env.sh sources a dataset preset; the conversion does not depend on which
      source scripts/lib/env.sh
      # the conversion writes this file last: a directory without it is a failed conversion, which convert_ckpt.sh
      # refuses to overwrite (remove it by hand)
      if [ -f "$MODEL_TORCH_DIST/latest_checkpointed_iteration.txt" ]; then echo "[$PRL_SITE] $MODEL_TORCH_DIST exists"; exit 0; fi
      in_miles python3 -c "import torch, sglang, megatron; print('container ok:', torch.__version__, torch.cuda.device_count(), 'GPUs')"
      bash scripts/setup/convert_ckpt.sh ) ;;
  smoke)
    : "${SMOKE_ARM:?the smoke step needs SMOKE_ARM}"
    [ "${ALGO:-grpo}" != ppo ] || { echo "the smoke step runs a one-job smoke; PPO: CHAIN=smoke_ppo" >&2; exit 1; }
    local found; found=$(find_smoke "${ARM:-$SMOKE_ARM}")
    if [ -n "$found" ]; then echo "[$PRL_SITE] $found already vouches for these sources and $MODEL at this context"; return 0; fi
    # the training step looks for a smoke at its own arm's context (grpo_full.sh requires the same)
    [ -z "${ARM:-}" ] || [ "$(arm_setting MAX_CONTEXT "$SMOKE_ARM")" = "$(arm_setting MAX_CONTEXT)" ] ||
      { echo "SMOKE_ARM=$SMOKE_ARM and ARM=$ARM have different contexts" >&2; exit 1; }
    sandbox
    smoke_contract "$SMOKE_ARM"
    start_workers "$SMOKE_ARM"
    env -u TRANSFORM ARM="$SMOKE_ARM" bash $T/grpo_smoke.sh
    PRL_SMOKE_JOB=$JOB_ID PRL_SMOKE_RUN=$(arm_setting RUN_DIR "$SMOKE_ARM") bash $T/grpo_acceptance.sh ;;
  smoke_grpo)
    sandbox
    smoke_contract "$ARM"
    start_workers "$ARM"
    bash $T/grpo_smoke.sh
    PRL_SMOKE_JOB=$JOB_ID bash $T/grpo_acceptance.sh ;;
  smoke_ppo)
    sandbox
    : "${PRL_RUN_ID:?the PPO smoke needs PRL_RUN_ID}"
    PRL_PPO_SMOKE_ROLLOUTS=2 bash $T/grpo_smoke.sh
    PRL_PPO_SMOKE_ROLLOUTS=3 PRL_SMOKE_RESUME_FROM=1 bash $T/grpo_smoke.sh
    PRL_SMOKE_JOB=$JOB_ID PRL_SMOKE_RUN=$PRL_EXT/runs/ppo_ota_train_smoke_$PRL_RUN_ID \
      bash $T/grpo_acceptance.sh ;;
  train_eval)
    sandbox
    : "${PRL_RUN_ID:?} ${ARM:?}"
    # the submit script names the smoke when the chain has no smoke step (PRL_SUBMIT_FINDS_SMOKE=1, the default);
    # otherwise the newest one now
    PRL_SMOKE_RUN=${PRL_SMOKE_RUN:-$(find_smoke)}
    [ -n "$PRL_SMOKE_RUN" ] || { echo "no accepted smoke for these sources and $MODEL at this context" >&2; exit 1; }
    export PRL_SMOKE_RUN
    start_workers "$ARM"
    bash $T/grpo_full.sh
    local run last latest iter out
    run=$(arm_setting RUN_DIR); last=$(($(arm_setting NUM_ROLLOUT) - 1))
    latest=$(cat "$run/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || echo -1)
    if [ "$latest" != "$last" ]; then echo "[$PRL_SITE] $run at iteration $latest of $last: resubmit to continue"; return 0; fi
    iter=$(printf 'iter_%07d' "$last"); out=$PRL_EXT/ckpt/hf/$(basename "$run")_$iter
    [ -f "$out/.verified" ] || [ -f "$out/config.json" ] || CKPT="$run/ckpt/$iter" OUT="$out" bash $T/ckpt_to_hf.sh
    MODEL_HF_DIR="$out" ARM=$EVAL_ARM bash $T/tbdev70_eval.sh ;;
  eval_base)
    sandbox
    env -u MODEL_HF_DIR ARM="$EVAL_ARM" bash $T/tbdev70_eval.sh ;;
  eval_hf)
    sandbox
    MODEL_HF_DIR=${MODEL_HF_DIR:?the exported checkpoint to evaluate} ARM="$EVAL_ARM" bash $T/tbdev70_eval.sh ;;
  unit)  # the unit tests in the Miles image (torch, Miles; a CPU-only venv skips those), on a compute node
    export DATASET=otagent HARNESS=harbor
    source scripts/lib/env.sh
    in_miles bash -c "cd $PRL_ROOT && python3 -m pytest -q -p no:cacheprovider tests/unit" ;;
  replay)  # train-only checks and failure drills (PRL_REPLAY_SOURCE: an accepted smoke's RUN_DIR; default this
    sandbox  # job's smoke_grpo run)
    PRL_REPLAY_SOURCE=${PRL_REPLAY_SOURCE:-$(arm_setting RUN_DIR)} bash $T/replay_check.sh ;;
  task_check)  # oracle / nop trials on a template snapshot (PRL_TEMPLATE_MANIFEST, PRL_TASK_ROOT), or the smoke's tasks
    sandbox
    if [ -n "${PRL_TEMPLATE_MANIFEST:-}" ]; then bash $T/task_smoke.sh; else smoke_contract ota_train_smoke; fi ;;
  raydebug)  # Ray head start as container_main.sh does it, with its logs kept under $PRL_EXT/raydebug_<job>
    sandbox
    export DATASET=otagent HARNESS=harbor
    source scripts/lib/env.sh
    local d=$PRL_EXT/raydebug_$JOB_ID; mkdir -p "$d"
    mkdir -p "$PRL_SCRATCH/plasma"
    in_miles env RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 bash -c "ray start --head --node-ip-address 127.0.0.1 --port 6379 \
      --num-gpus 4 --disable-usage-stats --include-dashboard=false --temp-dir $PRL_SCRATCH/r ${RAYDEBUG_ARGS//@SCRATCH@/$PRL_SCRATCH} \
      && ray status && python3 -c 'import ray; ray.init(address=\"auto\"); r = ray.put(b\"x\" * 10**8); print(\"put/get ok\", len(ray.get(r)))'; ray stop --force" || true
    cp -r "$PRL_SCRATCH/r" "$d/" 2>/dev/null || true   # AF_UNIX paths must stay under 107 bytes: run short, keep a copy
    for f in $(ls -d "$d"/r/session_*/logs 2>/dev/null | head -1)/{raylet.out,raylet.err,gcs_server.out,gcs_server.err}; do
      echo "=== $f"; tail -40 "$f" 2>/dev/null; done ;;
  *) echo "unknown step $1 in PRL_CHAIN=$PRL_CHAIN" >&2; exit 1 ;;
  esac
}

# archive_job_trials: pack the trial directories this job wrote (only paths that name its JOB_ID)
archive_job_trials() {
  local dirs=() d steps=" ${PRL_CHAIN//+/ } "
  if [[ $steps == *" train_eval "* ]] && [ -n "${ARM:-}" ]; then dirs+=("$(arm_setting HARBOR_TRIALS_DIR)"); fi
  if [[ $steps == *" train_eval "* || $steps == *" eval_base "* || $steps == *" eval_hf "* ]]; then
    dirs+=("$(arm_setting HARBOR_TRIALS_DIR "${EVAL_ARM:?}")")
  fi
  for d in "${dirs[@]}"; do [[ $d == *"$JOB_ID"* ]] && archive_trials "$d"; done
  return 0
}

run_chain() {
  mkdir -p "$PRL_EXT/logs" "$PRL_EXT/locks"
  exec > >(tee -a "$PRL_EXT/logs/${PRL_CHAIN:?}_$JOB_ID.log") 2>&1
  trap 'archive_job_trials' EXIT
  trap 'exit 143' TERM  # so that the EXIT trap also runs when the scheduler stops the job
  echo "[$PRL_SITE] job $JOB_ID chain $PRL_CHAIN on $(hostname) at $(date); scratch $PRL_SCRATCH"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  local s
  for s in ${PRL_CHAIN//+/ }; do
    echo "[$PRL_SITE] step $s at $(date)"
    run_step "$s"
  done
  echo "[$PRL_SITE] job $JOB_ID chain $PRL_CHAIN done at $(date)"
}
