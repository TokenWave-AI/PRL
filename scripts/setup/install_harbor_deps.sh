#!/bin/bash
# Installs Harbor (harbor-miles-v0.20.0 at a pinned commit) + the e2b SDK + litellm for the Miles container into
# $PRL_EXT/pydeps_harbor, put on PYTHONPATH by in_miles when HARNESS=harbor (scripts/lib/env.sh), and the e2b SDK
# version of the task templates into $PRL_EXT/pydeps_e2b_sdk.
# Only packages the container's /opt/sglang python lacks are installed (--no-deps, pinned): a full resolve would
# shadow the container's own openai/pydantic/protobuf/tokenizers, which SGLang and Miles depend on.
# Run on a node with internet: PRL_SITE=<site> bash scripts/setup/install_harbor_deps.sh (a compute node, not the login node)
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lib/env.sh
mkdir -p "${PRL_SCRATCH:-/tmp}"   # job-local scratch may not exist yet
module load apptainer >/dev/null 2>&1 || true
T=$PRL_EXT/pydeps_harbor
SRC=$(mktemp -d "${PRL_SCRATCH:-/tmp}/harbor_src.XXXX")
# The branch harbor-miles-v0.20.0 keeps moving: one push rewrote its E2B environment's exec, which
# prl/rollout/harbor_fixes.py patches. Pin the commit, so every site runs the same Harbor.
HARBOR_COMMIT=917b3c1be7cf6553a7a9fe25de8adb6f8e639287
git clone -q --filter=blob:none -b harbor-miles-v0.20.0 https://github.com/harbor-framework/harbor.git "$SRC/harbor"
git -C "$SRC/harbor" checkout -q "$HARBOR_COMMIT"
PKGS="bracex==3.1 connectrpc==0.11.1 deprecation==2.1.0 dirhash==0.5.0 e2b==2.56.0 e2b_dockerfile_utils==0.1.1
dockerfile_parse==2.0.1 fastuuid==0.14.0 litellm==1.105.1 postgrest==2.32.0 protobuf_py==0.1.1 protobuf_py_ext==0.1.1 pydantic_settings==2.16.0
pyqwest==0.12.0 realtime==2.32.0 scantree==0.0.4 shortuuid==1.0.13 storage3==2.32.0 StrEnum==0.4.15 supabase==2.32.0
supabase_auth==2.32.0 supabase_functions==2.32.0 tenacity==9.2.1 toml==0.10.2 typed_argument_parser==1.12.0 wcmatch==11.1"
PKGS=$(echo $PKGS)   # one line: the list above is multi-line, and bash -c would split the command at the newlines
apptainer exec -B "$PRL_EXT" -B "$SRC" "$MILES_IMAGE" bash -c "cd / && /opt/sglang/bin/python3 -m pip install -q --no-deps --target $T $PKGS $SRC/harbor && \
  PYTHONPATH=$T /opt/sglang/bin/python3 -c 'import harbor, e2b, litellm; from harbor.environments.e2b import E2BEnvironment; from harbor.agents.terminus_2.terminus_2 import Terminus2; print(\"harbor deps ok\")'"
rm -rf "$SRC"
# The e2b SDK the task templates were checked with, ahead of pydeps_harbor's on the jobs' PYTHONPATH
# (PRL_EXTRA_PYTHONPATH in configs/arms/common_ota.env)
apptainer exec -B "$PRL_EXT" "$MILES_IMAGE" /opt/sglang/bin/python3 -m pip install -q --disable-pip-version-check \
  --no-cache-dir --no-deps --target "$PRL_EXT/pydeps_e2b_sdk" e2b==2.50.0 pyqwest==0.10.0
