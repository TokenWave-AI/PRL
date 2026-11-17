#!/bin/bash
# Chain step eval (scripts/lib/chain.sh: eval_base, train_eval, eval_hf): TB-dev 70 x EVAL_N_SAMPLES for any model
# profile (configs/models/) and eval arm (default ota_eval_tbdev70_64k_summary_c16_noturncap); checks that the arm is
# self-consistent. A trained checkpoint: MODEL_HF_DIR=<hf dir from scripts/chain/ckpt_to_hf.sh>.
set -euo pipefail
export PRL_ROOT=${PRL_ROOT:-$PWD} PRL_SITE=${PRL_SITE:?set PRL_SITE (configs/sites)}
: "${ARM:?set ARM=<eval arm in configs/arms>}"
# A passed MODEL_HF_DIR must be a verified export: a dependent job may start whatever the conversion's exit status,
# so a failed or half-written conversion has to stop here.
if [ -n "${MODEL_HF_DIR:-}" ]; then
  [ -s "$MODEL_HF_DIR/export_verified.json" ] || { echo "[eval] $MODEL_HF_DIR is not a verified export" >&2; exit 1; }
  cat "$MODEL_HF_DIR/export_verified.json"
fi
set -a
source "$PRL_ROOT/configs/sites/$PRL_SITE.env"
source "$PRL_ROOT/configs/arms/common_ota_single_node.env"
source "$PRL_ROOT/configs/arms/$ARM.env"
set +a
source "$PRL_ROOT/scripts/lib/env.sh"   # model profile (MODEL_*); run_job.sh sources it again
echo "[eval] MODEL=$MODEL model $MODEL_HF_DIR"
[ -f "$MODEL_HF_DIR/config.json" ] || { echo "[eval] no HF checkpoint at '$MODEL_HF_DIR'" >&2; exit 1; }
PYTHONPATH="$PRL_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PRL_HOST_PYTHON" - <<'PY'
import json
import os
import shlex
from pathlib import Path
from prl.sandbox.template_manifest import resolve_template
env = os.environ
rows = [json.loads(line) for line in Path(env['EVAL_DATA']).read_text().splitlines()]
tasks = [row['metadata']['instance_id'] for row in rows]
manifest_path = env['PRL_TEMPLATE_MANIFEST']
manifest = json.loads(Path(manifest_path).read_text())
assert len(tasks) == len(set(tasks)) == 70, 'Expected all 70 TB-dev tasks'
assert 'png-generation' in tasks, 'Full evaluation must include png-generation'
assert set(tasks) == set(manifest['tasks']), 'Dataset and template manifest differ'
for task in tasks:
    resolve_template(manifest_path, Path(env['HARBOR_TASKS_DIR']) / task, env['E2B_API_URL'])
# Context budgets agree with each other and fit the model (no rope scaling is configured).
context = int(env['MAX_CONTEXT'])
assert env['HARBOR_MAX_SEQ_LEN'] == env['AGENT_MAX_INPUT_TOKENS'] == str(context), 'context settings must agree'
assert env['MAX_RESPONSE'] == env['AGENT_MAX_OUTPUT_TOKENS'] and int(env['MAX_RESPONSE']) < context
assert context <= int(env['MODEL_MAX_CONTEXT']), (context, env['MODEL_MAX_CONTEXT'])
assert env['MODE'] == 'rollout' and env['NUM_ROLLOUT'] == '0'
assert env['NUM_GPUS'] == env['ACTOR_GPUS_PER_NODE'] == env['ROLLOUT_GPUS']
assert env['NNODES'] == env['COLOCATE'] == '1'
extra = shlex.split(env['EXTRA_ARGS'])
assert extra[extra.index('--rollout-num-gpus-per-engine') + 1] == env['ROLLOUT_GPUS']
assert extra[extra.index('--use-session-server') + 1] == 'v2'
assert extra[extra.index('--session-sample-picker-path') + 1] == 'prl.rollout.eval_sample_picker.latest_eval_leaf'
from prl.rollout.eval_sample_picker import latest_eval_leaf
assert latest_eval_leaf([], {}) == []
concurrency = int(extra[extra.index('--sglang-server-concurrency') + 1])
assert json.loads(env['HARBOR_ENV_KWARGS'])['prl_max_concurrent_trials'] == concurrency
n = int(env['EVAL_N_SAMPLES'])
print(f"70/70 ready templates; {70 * n} trials (x{n}); {env['MODEL']}; context={context}; "
      f"summarize={env['HARBOR_TERMINUS_2_ENABLE_SUMMARIZE']}; TITO={env['TITO_MODEL']} v2; "
      f"{env['NUM_GPUS']} GPUs / TP={env['ROLLOUT_GPUS']} / concurrency={concurrency}", flush=True)
PY
export COMMON=common_ota_single_node ARM
exec bash "$PRL_ROOT/scripts/run_job.sh"
