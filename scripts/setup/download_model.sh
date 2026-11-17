#!/bin/bash
# Download a model profile's weights (configs/models/<MODEL>.env: MODEL_HF_REPO) into $HF_HOME/hub with the Miles
# container's `hf`, then print the snapshot the profile resolves. Needs internet on the node (a site proxy is fine);
# usually as the chain step download: CHAIN=download MODEL=<profile> bash scripts/<site>/submit.sh
set -euo pipefail
source "$(dirname "$0")/../lib/env.sh"
in_miles hf download "$MODEL_HF_REPO" >/dev/null
unset MODEL_HF_DIR; set -a; source "$PRL_ROOT/configs/models/$MODEL.env"; set +a   # re-resolve after the download
[ -f "$MODEL_HF_DIR/config.json" ] || { echo "MODEL=$MODEL: '$MODEL_HF_REPO' did not land in $HF_HOME/hub" >&2; exit 1; }
echo "[download_model] $MODEL_HF_REPO -> $MODEL_HF_DIR"; ls -la "$MODEL_HF_DIR"
