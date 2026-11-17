#!/bin/bash
# HF -> Megatron torch_dist (needed as --ref-load for training) for a model profile (configs/models/<MODEL>.env;
# MODEL unset = qwen3.5-$MODEL_SIZE as before). Usage: MODEL=qwen3-8B bash scripts/setup/convert_ckpt.sh
# On a GPU node; usually as the chain step convert (scripts/lib/chain.sh), which runs it once per model:
#   CHAIN=download+convert MODEL=<profile> bash scripts/<site>/submit.sh
set -euo pipefail
source "$(dirname "$0")/../lib/env.sh"
[ -f "$MODEL_HF_DIR/config.json" ] || { echo "MODEL=$MODEL: no HF checkpoint at '$MODEL_HF_DIR' (download $MODEL_HF_REPO)" >&2; exit 1; }
[ -e "$MODEL_TORCH_DIST" ] && { echo "$MODEL_TORCH_DIST exists; refusing to overwrite" >&2; exit 1; }
mkdir -p "$(dirname "$MODEL_TORCH_DIST")"
in_miles bash -c "cd $PRL_ROOT/build/miles && read -ra MODEL_ARGS <<< \"\$(python3 miles/utils/external_utils/model_args_utils.py $MODEL_ARGS_NAME)\" && \
  python3 tools/convert_hf_to_torch_dist.py \${MODEL_ARGS[@]} --hf-checkpoint $MODEL_HF_DIR --save $MODEL_TORCH_DIST"
ls -la "$MODEL_TORCH_DIST"
