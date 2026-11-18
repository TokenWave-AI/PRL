#!/bin/bash
# Chain step train_eval, training part (scripts/lib/chain.sh): full OTA training relayed over consecutive jobs. All jobs
# share PRL_RUN_ID, so they share RUN_DIR; each successor resumes from $RUN_DIR/ckpt. A job finding the run
# complete exits at once; one finding no new checkpoint since the previous job stops (scripts/lib/relay.sh).
# PRL_SMOKE_RUN names the accepted smoke whose receipt must match these sources (scripts/checks/find_smoke.py).
# ARM: e.g. ota_train_transform with TRANSFORM=<name>[, ALGO=ppo]; other model / context: MODEL=qwen3-8B with a
# 40k arm and a 40k smoke. PPO: the critic resumes from $RUN_DIR/ckpt_critic (critic_load in scripts/lib/relay.sh).
# Every job checks the transform on recorded trials before training (scripts/lib/transform_check.sh); the smoke receipt
# vouches for the framework sources only, so a new transform needs no smoke.
# PRL_FULL_WITHOUT_ACCEPTANCE=<reason>: start a fresh run (no checkpoint yet) from a smoke that ran but did not
# pass acceptance; PRL_SMOKE_RUN still names it, for the settings checks. Successor jobs then refuse it.
set -euo pipefail
export PRL_ROOT=${PRL_ROOT:-$PWD} PRL_SITE=${PRL_SITE:?set PRL_SITE (configs/sites)}
set -a
source "$PRL_ROOT/configs/sites/$PRL_SITE.env"
source "$PRL_ROOT/configs/arms/common_ota_single_node.env"
ARM=${ARM:-ota_train_full}
source "$PRL_ROOT/configs/arms/$ARM.env"
set +a
source "$PRL_ROOT/scripts/lib/env.sh"   # model profile (MODEL_*); run_job.sh sources it again
source "$PRL_ROOT/scripts/lib/relay.sh"
relay_claim "$RUN_DIR"
status=0
relay_begin "$RUN_DIR" "$NUM_ROLLOUT" || status=$?
[ "$status" = 3 ] && exit 0
[ "$status" = 0 ] || exit "$status"
mkdir -p "$HARBOR_TRIALS_DIR"
source "$PRL_ROOT/scripts/lib/guard.sh"
hardware_gate
PYTHONPATH="$PRL_ROOT:$PRL_ROOT/scripts/checks${PYTHONPATH:+:$PYTHONPATH}" "$PRL_HOST_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from prl.sandbox.template_manifest import resolve_template
from grpo_acceptance import source_hashes
from find_smoke import mismatch
smoke = Path(os.environ.get('PRL_SMOKE_RUN') or Path(os.environ['PRL_EXT']) / 'runs' / (
    'grpo_ota_train_smoke_' + os.environ['PRL_SMOKE_JOB']))
if os.environ.get('PRL_FULL_WITHOUT_ACCEPTANCE'):
    # Start without an accepted smoke, the reason standing in for its receipt (e.g. a fresh PPO run while a known bug
    # breaks the weights when the PPO smoke resumes). The smoke's settings are still checked below. Fresh runs only:
    # resuming is what that smoke failed.
    assert not (Path(os.environ['RUN_DIR']) / 'ckpt' / 'latest_checkpointed_iteration.txt').exists(), \
        'PRL_FULL_WITHOUT_ACCEPTANCE cannot resume a run'
    receipt = {'passed': False, 'without_acceptance': os.environ['PRL_FULL_WITHOUT_ACCEPTANCE'],
               'algorithm': os.environ.get('ADVANTAGE_ESTIMATOR', 'grpo')}
else:
    receipt = json.loads((smoke / 'acceptance.json').read_text())
algorithm = os.environ.get('ADVANTAGE_ESTIMATOR', 'grpo')
# The smoke vouches for these sources, this algorithm, model, context and schedule: scripts/checks/find_smoke.py, which
# also picked it (without an acceptance, everything but the receipt itself)
why = mismatch(smoke, {**receipt, 'passed': True, 'source_hashes': source_hashes(Path(os.environ['PRL_ROOT']))}
               if receipt.get('without_acceptance') else receipt, source_hashes(Path(os.environ['PRL_ROOT'])),
               algorithm, os.environ['MODEL'], os.environ['MAX_CONTEXT'], os.environ.get('ASYNC', '0'))
assert not why, f'{smoke} does not vouch for this job: {why}'
smoke_settings = json.loads((smoke / 'preflight.json').read_text())['settings']
ppo = ('CRITIC_LR', 'GAE_GAMMA', 'GAE_LAMBDA', 'NORMALIZE_ADVANTAGES')
if algorithm == 'ppo':
    assert all(smoke_settings[key] == os.environ[key] for key in ppo), {key: smoke_settings[key] for key in ppo}
    assert 0 < int(os.environ['CRITIC_WARMUP_STEPS']) < int(os.environ['NUM_ROLLOUT'])
# The trajectory transform is not the smoke's business: this job checks it on recorded trials before training (below,
# in the Miles container: the user's code runs there, not with this host Python)
transform = os.environ.get('PRL_TRANSFORM') or 'none'
context = int(os.environ['MAX_CONTEXT'])
assert os.environ['HARBOR_MAX_SEQ_LEN'] == os.environ['AGENT_MAX_INPUT_TOKENS'] == str(context)
assert os.environ['MAX_RESPONSE'] == os.environ['AGENT_MAX_OUTPUT_TOKENS'] and int(os.environ['MAX_RESPONSE']) < context
assert context <= int(os.environ['MODEL_MAX_CONTEXT'])
(Path(os.environ['RUN_DIR']) / 'smoke_acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
def tasks(path):
    return [json.loads(line)['metadata']['instance_id'] for line in Path(path).read_text().splitlines()]
train, evaluation = tasks(os.environ['PROMPT_DATA']), tasks(os.environ['TBDEV_DATA'])
assert len(train) == len(set(train)) == 728
assert len(evaluation) == len(set(evaluation)) == 70
assert not set(train).intersection(evaluation)
manifest = os.environ['PRL_TEMPLATE_MANIFEST']
# Every training task (and the tasks of an evaluation during training, EVAL_DATA) has a ready template in the training
# snapshot; TB-dev's are the evaluation snapshot's
needed = train + (tasks(os.environ['EVAL_DATA']) if os.environ.get('EVAL_DATA') else [])
missing = sorted(set(needed) - set(json.loads(Path(manifest).read_text())['tasks']))
assert not missing, (f'{len(missing)} tasks have no template in {manifest}, e.g. {missing[:3]}: '
                     'build them with scripts/setup/build_sandboxes.sh (docs/sandboxes.md)')
for task in needed:
    resolve_template(manifest, Path(os.environ['HARBOR_TASKS_DIR']) / task, os.environ['E2B_API_URL'])
# The trainer computes the sampled tokens' log-probs at temperature 1 over the full vocabulary: other sampling settings
# would make the train/rollout log-prob comparison (and the importance ratios) wrong without an error
assert os.environ.get('TEMPERATURE', '1.0') == os.environ.get('TOP_P', '1.0') == '1.0'
# The data a resumed run continues with is the data it started with: prompts,
# templates, tokenizer, order. The first job records it; a change stops the run (delete the file to accept one).
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
model = Path(os.environ['MODEL_HF_DIR'])
fingerprint = {'PROMPT_DATA': sha(os.environ['PROMPT_DATA']), 'PRL_TEMPLATE_MANIFEST': sha(manifest),
               'HARBOR_TASKS_DIR': os.environ['HARBOR_TASKS_DIR'], 'ROLLOUT_SEED': os.environ.get('ROLLOUT_SEED', '42'),
               'tokenizer': {f.name: sha(f) for f in sorted(model.glob('*token*')) + sorted(model.glob('chat_template*'))}}
recorded = Path(os.environ['RUN_DIR']) / 'data_fingerprint.json'
if recorded.exists():
    before = json.loads(recorded.read_text())
    assert before == fingerprint, ('the data changed since this run started: '
                                   + str({k: (before.get(k), v) for k, v in fingerprint.items() if before.get(k) != v})
                                   + f'; delete {recorded} to accept it')
else:
    recorded.write_text(json.dumps(fingerprint, indent=2) + '\n')
assert os.environ['NUM_GPUS'] == '4' and os.environ['ACTOR_GPUS_PER_NODE'] == os.environ['ROLLOUT_GPUS'] == '2'
assert os.environ['ASYNC'] == '0'
assert not os.environ.get('EVAL_DATA'), 'TB-dev is evaluated after training'
assert os.environ['PRL_REJECT_MISMATCH'] in ('0', '1')
assert int(os.environ['PRL_MISMATCH_MAX_RETRIES']) >= 0
extra = os.environ['EXTRA_ARGS'].split()
concurrency = int(extra[extra.index('--sglang-server-concurrency') + 1])
assert json.loads(os.environ['HARBOR_ENV_KWARGS'])['prl_max_concurrent_trials'] == concurrency
schedule = (f"{os.environ['NUM_ROLLOUT']} rollouts, the first {os.environ['CRITIC_WARMUP_STEPS']} critic only; "
            + ', '.join(f"{key} {os.environ[key]}" for key in ppo) if algorithm == 'ppo'
            else f"{os.environ['NUM_ROLLOUT']} updates")
print(f"Full training inputs ready: {os.environ['MODEL']} ({os.environ['MODEL_HF_DIR']}), context {context}; "
      f"728 train (TB-dev after training); {algorithm}: {schedule}; "
      f"{os.environ['ROLLOUT_BATCH']}x{os.environ['N_SAMPLES']}; "
      f"{concurrency} concurrent trials; reject_mismatch={os.environ['PRL_REJECT_MISMATCH']}; transform {transform}"
      + (f" (judge {os.environ.get('PRL_JUDGE_MODEL')}, then {os.environ.get('PRL_JUDGE_FALLBACK_MODELS') or '-'}"
         f" on a refusal, {os.environ.get('PRL_JUDGE_EXTRA_BODY')}); "
         if transform != 'none' else "; ") +
      "2 train + 2 inference GPUs", flush=True)
PY
source "$PRL_ROOT/scripts/lib/transform_check.sh"
transform_check
# Nothing is saving yet: also drop a partial checkpoint left by a job that reached its time limit mid-save.
KEEP_ITERS="$PRL_KEEP_ITERS" bash "$PRL_ROOT/scripts/tools/prune_ckpts.sh" "$RUN_DIR"
# One checkpoint per update would otherwise fill ~117 GB per step: prune once a minute while training
run_guarded "full training" 'LIVE=1 KEEP_ITERS="$PRL_KEEP_ITERS" bash "$PRL_ROOT/scripts/tools/prune_ckpts.sh" "$RUN_DIR" >> "$RUN_DIR/prune.log" 2>&1'
