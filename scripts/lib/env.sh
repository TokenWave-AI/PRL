# Shared paths + container helper. Source me. The site profile (PRL_SITE -> configs/sites/<site>.env, see
# configs/sites/example.env) sets PRL_EXT, PRL_SCRATCH, HF_HOME and MILES_IMAGE; without one, set them yourself.
export PRL_ROOT=${PRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
# Site profile (paths, scheduler, container): PRL_SITE=<site> -> configs/sites/<site>.env
if [ -n "${PRL_SITE:-}" ]; then
  [ -f "$PRL_ROOT/configs/sites/$PRL_SITE.env" ] || { echo "unknown PRL_SITE=$PRL_SITE" >&2; return 1 2>/dev/null || exit 1; }
  set -a; source "$PRL_ROOT/configs/sites/$PRL_SITE.env"; set +a
fi
# The site profile sets these (configs/sites/<site>.env); nothing here guesses a path
: "${PRL_EXT:?set PRL_SITE (configs/sites) or PRL_EXT}" "${PRL_SCRATCH:?set PRL_SITE (configs/sites) or PRL_SCRATCH}"
: "${HF_HOME:?set PRL_SITE (configs/sites) or HF_HOME}" "${MILES_IMAGE:?set PRL_SITE (configs/sites) or MILES_IMAGE}"
export PRL_EXT PRL_SCRATCH HF_HOME MILES_IMAGE
# Extra Python packages for the container (pinned deps installed once per site)
export PRL_PYDEPS=${PRL_PYDEPS:-$PRL_EXT/pydeps_miles}
export PRL_EXTRA_BINDS=${PRL_EXTRA_BINDS:-}
export QWEN35_9B=${QWEN35_9B:-$(ls -d $HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/* 2>/dev/null | head -1)}
# Model profile: MODEL=<name> -> configs/models/<name>.env (HF dir, Megatron args, ref checkpoint, TITO template, SGLang
# parsers, max context; see its README). Unset: qwen3.5-$MODEL_SIZE (launches that set only MODEL_SIZE), default 9B.
export MODEL=${MODEL:-qwen3.5-${MODEL_SIZE:-9B}}
[ -f "$PRL_ROOT/configs/models/$MODEL.env" ] || { echo "unknown MODEL=$MODEL; one of: $(cd "$PRL_ROOT/configs/models" && ls *.env | sed 's/\.env$//' | xargs)" >&2; return 1 2>/dev/null || exit 1; }
set -a; source "$PRL_ROOT/configs/models/$MODEL.env"; set +a
# Dataset preset -> configs/datasets/$DATASET.env (otagent: OpenThoughts-Agent-v1-RL, Harbor)
export DATASET=${DATASET:-otagent}
# Log the policy entropy of each training step (Miles --observe-training-entropy; with PRL_FUSED_LOGPROB=1 the
# fused kernel computes it, prl/algo/fused_logprob.py). Exported here so worker.env carries it to the trainer ranks.
export PRL_OBSERVE_ENTROPY=${PRL_OBSERVE_ENTROPY:-1}
[ -f "$PRL_ROOT/configs/datasets/$DATASET.env" ] || { echo "unknown DATASET=$DATASET" >&2; return 1 2>/dev/null || exit 1; }
set -a; source "$PRL_ROOT/configs/datasets/$DATASET.env"; set +a

# (Re)build build/miles if patches changed and pin this launch to the resolved versioned tree (build/miles@<stamp>),
# so later rebuilds never change code under a running job. Multi-node workers inherit the head's PRL_MILES_DIR via
# worker.env and skip the build (PRL_SKIP_APPLY_PATCHES=1).
resolve_build() {
  if [ "${PRL_SKIP_APPLY_PATCHES:-0}" != 1 ] && [ -z "${PRL_MILES_DIR:-}" ]; then
    bash "$PRL_ROOT/scripts/setup/apply_patches.sh" >&2 || return 1
  fi
  export PRL_MILES_DIR=${PRL_MILES_DIR:-$(readlink -f "$PRL_ROOT/build/miles")}
}

# Run a command inside the Miles container with GPUs, our repo and data. The PATCHED tree in build/ (= vendored
# upstream + patches/, see scripts/setup/apply_patches.sh) shadows the container's /root/miles. Agent sandboxes are
# E2B sandboxes (Harbor), not local containers.
in_miles() {
  resolve_build || return 1
  mkdir -p "$PRL_SCRATCH/home" "${APPTAINER_CONFIGDIR:-$PRL_SCRATCH/apptainer_cfg}"
  # Own $HOME so the host ~/.bashrc / ~/.local never leak into the container.
  # --no-mount: a site may bind host /opt (shadows the image venv /opt/sglang) and hostfs into every container.
  # Bind only what we use: repo, external artifacts, HF cache, scratch (+ PRL_EXTRA_BINDS, space-separated).
  local binds=() d
  # HARNESS=harbor: Harbor + e2b + litellm (scripts/setup/install_harbor_deps.sh) and Miles' harbor example modules
  [ "${HARNESS:-}" = harbor ] && PRL_EXTRA_PYTHONPATH="${PRL_EXTRA_PYTHONPATH:+$PRL_EXTRA_PYTHONPATH:}$PRL_EXT/pydeps_harbor:$PRL_MILES_DIR/examples/experimental/harbor:$PRL_MILES_DIR/examples/swe-agent-harbor-docker"
  for d in "$PRL_ROOT" "$PRL_EXT" "$HF_HOME" "$PRL_SCRATCH" ${PRL_EXTRA_BINDS:-}; do [ -e "$d" ] && binds+=(-B "$d"); done
  # PRL_MILES_EXTRA_ARGS: site-specific `apptainer exec` flags for the Miles container (--pid, see configs/sites)
  apptainer exec --nv --cleanenv --no-mount hostfs,bind-paths --home "$PRL_SCRATCH/home" ${PRL_MILES_EXTRA_ARGS:-} \
    "${binds[@]}" -B "$HOME/.ssh:$PRL_SCRATCH/home/.ssh:ro" -B /etc/hosts -B /etc/resolv.conf \
    --env PYTHONPATH="$PRL_ROOT:$PRL_MILES_DIR:$PRL_PYDEPS${PRL_EXTRA_PYTHONPATH:+:$PRL_EXTRA_PYTHONPATH}:/root/Megatron-LM" \
    --env PRL_MILES_DIR="$PRL_MILES_DIR",HF_HOME="$HF_HOME",APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-}" \
    "$MILES_IMAGE" "$@"
}
