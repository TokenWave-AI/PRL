#!/bin/bash
# Chain step convert (scripts/lib/chain.sh): Megatron torch_dist checkpoint -> verified HF directory for evaluation
# (scripts/tools/torch_dist_to_hf.py). CPU only: the 9B export holds ~18 GB of model tensors plus one 5 GB shard
# (optimizer state is not read). Reads the checkpoint only: safe while the run is still training, as long as the
# iteration is kept (PRL_KEEP_ITERS) or is the latest. CKPT=$PRL_EXT/runs/<run>/ckpt/iter_<n>, OUT=<hf dir>.
# ROUNDTRIP=1 (default) first exports the base torch_dist (the run's starting point) in check-only mode and requires
# it to be bitwise equal to the base HF model: that proves the name/shape mapping before trusting a trained export.
# PRL_MILES_DIR: the build tree that trained the checkpoint (default: the current build/miles, which is what a run
# pins unless patches changed since). MODEL: the profile the run trained (configs/models/; default qwen3.5-9B): its
# MODEL_HF_DIR is the base, its MODEL_TORCH_DIST/release the round-trip input.
set -euo pipefail
export PRL_ROOT=${PRL_ROOT:-$PWD} PRL_SITE=${PRL_SITE:?set PRL_SITE (configs/sites)} DATASET=${DATASET:-otagent} MODEL=${MODEL:-qwen3.5-9B}
set -a; source "$PRL_ROOT/configs/sites/$PRL_SITE.env"; set +a
source "$PRL_ROOT/scripts/lib/env.sh"
: "${CKPT:?set CKPT=<run>/ckpt/iter_N}" "${OUT:?set OUT=<hf output dir>}"
export PRL_MILES_DIR=${PRL_MILES_DIR:-$(readlink -f "$PRL_ROOT/build/miles")}
[ -f "$CKPT/.metadata" ] || { echo "[ckpt_to_hf] no torch_dist checkpoint at $CKPT" >&2; exit 1; }
[ -e "$OUT" ] && { echo "[ckpt_to_hf] $OUT exists" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")" "$PRL_SCRATCH"
echo "[ckpt_to_hf] job ${JOB_ID:-?} on $(hostname): $CKPT -> $OUT (miles $PRL_MILES_DIR, $MODEL base $MODEL_HF_DIR)"
TOOL=$PRL_ROOT/scripts/tools/torch_dist_to_hf.py
if [ "${ROUNDTRIP:-1}" = 1 ]; then
  in_miles python3 "$TOOL" --input-dir "$MODEL_TORCH_DIST/release" \
    --origin-hf-dir "$MODEL_HF_DIR" --output-dir "$OUT.roundtrip" --expect-identical --check-only
fi
in_miles python3 "$TOOL" --input-dir "$CKPT" --origin-hf-dir "$MODEL_HF_DIR" --output-dir "$OUT"
du -sh "$OUT"
