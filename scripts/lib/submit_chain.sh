# The variables of a chain job (scripts/lib/chain.sh), shared by the scheduler wrappers scripts/pjm/submit.sh and
# scripts/pbs/submit.sh. Sourced from the repo root after configs/sites/$PRL_SITE.env and scripts/lib/chain.sh; reads
# CHAIN, MODEL, ALGO, ARM, SMOKE_ARM, EVAL_ARM, MODEL_HF_DIR, TRANSFORM, PRL_TRANSFORM_ARGS, PRL_RUN_ID,
# PRL_TEMPLATE_MANIFEST + PRL_TASK_ROOT [+ PRL_TASK_PROMPTS, PRL_TASK_CHECK_OUTPUT] (task_check; none: the smoke's tasks),
# PRL_REPLAY_SOURCE (replay),
# ELAPSE, RAYDEBUG_ARGS and sets vars (VAR=value,... for pjsub -x / qsub -v), elapse, nodes (the arms' NNODES) and record (the
# submission's record dir).
# submit_job COMMAND... runs the scheduler command, or prints it with DRY_RUN=1, and records it.
CHAIN=${CHAIN:?one step or several joined by +: prepare unit download convert smoke smoke_grpo smoke_ppo train_eval eval_base eval_hf task_check replay raydebug}
export ALGO=${ALGO:-grpo} MODEL=${MODEL:-qwen3.5-9B}
steps=" ${CHAIN//+/ } "
has() { [[ $steps == *" $1 "* ]]; }
for s in $steps; do
  case $s in prepare|unit|download|convert|smoke|smoke_grpo|smoke_ppo|train_eval|eval_base|eval_hf|task_check|replay|raydebug) ;;
    *) echo "unknown step $s" >&2; exit 1 ;; esac
done
rest=${steps/ unit / }; rest=${rest/ task_check / }; rest=${rest/ replay / }; rest=${rest/ prepare / }
rest=${rest/ download / }; rest=${rest/ convert / }
if { has smoke_grpo || has smoke_ppo; } && [ "$(wc -w <<< "$rest")" != 1 ]; then
  echo "smoke_grpo and smoke_ppo run alone, or with prepare, download, convert, unit, task_check and replay (they set ARM to the transform smoke)" >&2; exit 1
fi
vars=PRL_SITE=$PRL_SITE,PRL_CHAIN=$CHAIN,MODEL=$MODEL
# relay overrides of a resubmitted training chain (scripts/lib/relay.sh)
for v in PRL_RELAY_RESET PRL_RESUME_DATA_FROM_PREVIOUS; do [ -n "${!v:-}" ] && vars=$vars,$v=${!v}; done
# a chain of transforms as "a+b": pjsub -x and qsub -v split the variables on commas (prl/transforms/core.py
# reads both)
[[ "${TRANSFORM:-}" != *,* ]] || { echo "TRANSFORM=$TRANSFORM: write a chain as ${TRANSFORM//,/+}" >&2; exit 1; }
elapse=02:00:00
if has prepare; then elapse=04:00:00; fi  # the Harbor clone and the dataset download
record=$PRL_EXT/submissions/${PRL_SITE}_${CHAIN//+/_}_${MODEL}_$(date -u +%Y%m%d%H%M)
[ -n "${RAYDEBUG_ARGS:-}" ] && vars=$vars,RAYDEBUG_ARGS=$RAYDEBUG_ARGS
[ -n "${PRL_ARCHIVE_TRIALS:-}" ] && vars=$vars,PRL_ARCHIVE_TRIALS=$PRL_ARCHIVE_TRIALS  # 0: keep trial directories (chain.sh)
if has smoke_grpo || has smoke_ppo; then
  # ARM: an arm built on the transform smoke (e.g. ota_train_smoke_reroll: the post-process a reroll needs)
  vars=$vars,ARM=${ARM:-ota_train_smoke_transform},TRANSFORM=${TRANSFORM:?the smoke runs a transform}
  if has smoke_ppo; then vars=$vars,ALGO=ppo,PRL_RUN_ID=${PRL_RUN_ID:?}; fi
  # the transform's parameters too, when given (a smoke of a new code path of it, e.g. rewrite's mask_new)
  [ -n "${PRL_TRANSFORM_ARGS:-}" ] && vars=$vars,PRL_TRANSFORM_ARGS_FILE=$record/transform_args.json
  elapse=08:00:00
fi
if has smoke; then
  : "${SMOKE_ARM:?the smoke step needs SMOKE_ARM}"
  [ "$ALGO" != ppo ] || { echo "the smoke step runs a one-job smoke; PPO: CHAIN=smoke_ppo" >&2; exit 1; }
  vars=$vars,SMOKE_ARM=$SMOKE_ARM
  has train_eval || vars=$vars,ALGO=$ALGO  # the smoke looks for a receipt of this algorithm (REINFORCE++: its own)
  elapse=08:00:00
fi
if has eval_base || has train_eval || has eval_hf; then
  [ -n "${EVAL_ARM:-}" ] && vars=$vars,EVAL_ARM=$EVAL_ARM
  elapse=08:00:00
fi
if has task_check && [ -n "${PRL_TEMPLATE_MANIFEST:-}${PRL_TASK_ROOT:-}" ]; then  # neither: the smoke's 8 tasks (chain.sh)
  vars=$vars,PRL_TEMPLATE_MANIFEST=${PRL_TEMPLATE_MANIFEST:?task_check checks PRL_TEMPLATE_MANIFEST}
  vars=$vars,PRL_TASK_ROOT=${PRL_TASK_ROOT:?task_check needs PRL_TASK_ROOT, its task snapshot}
  # only a prompt file's tasks, written where the smoke preflight looks (scripts/chain/task_smoke.sh)
  for v in PRL_TASK_PROMPTS PRL_TASK_CHECK_OUTPUT; do [ -n "${!v:-}" ] && vars=$vars,$v=${!v}; done
fi
if has replay; then  # the smoke whose batches are replayed (scripts/chain/replay_check.sh): this job's own smoke
  if [ -n "${PRL_REPLAY_SOURCE:-}" ]; then vars=$vars,PRL_REPLAY_SOURCE=$PRL_REPLAY_SOURCE
  elif ! has smoke_grpo; then echo "replay needs PRL_REPLAY_SOURCE (a smoke RUN_DIR) or a smoke_grpo step" >&2; exit 1; fi
  if has smoke_grpo; then elapse=08:00:00; else elapse=04:00:00; fi
fi
if has eval_hf; then  # its name in the record: the checkpoint evaluated
  vars=$vars,MODEL_HF_DIR=${MODEL_HF_DIR:?eval_hf evaluates MODEL_HF_DIR, an exported checkpoint}
  record=$PRL_EXT/submissions/${PRL_SITE}_eval_hf_$(basename "$MODEL_HF_DIR")_$(date -u +%Y%m%d%H%M)
fi
if has train_eval; then
  : "${ARM:?} ${PRL_RUN_ID:?}"
  record=$PRL_EXT/submissions/${PRL_SITE}_$PRL_RUN_ID
  vars=$vars,ARM=$ARM,ALGO=$ALGO,PRL_RUN_ID=$PRL_RUN_ID
  if has smoke; then
    [ "$(arm_setting MAX_CONTEXT "$SMOKE_ARM")" = "$(arm_setting MAX_CONTEXT)" ] ||
      { echo "SMOKE_ARM=$SMOKE_ARM and ARM=$ARM have different contexts" >&2; exit 1; }
  elif [ "${PRL_SUBMIT_FINDS_SMOKE:-1}" = 1 ]; then  # else the job looks (the login node cannot run the host Python)
    smoke=$(find_smoke)
    [ -n "$smoke" ] || { echo "no accepted $ALGO smoke for these sources, $MODEL at ARM's context: add a smoke step$(
      [ "$ALGO" = ppo ] && echo " (PPO: CHAIN=smoke_ppo first, CONTRIBUTING.md)")" >&2; exit 1; }
    vars=$vars,PRL_SMOKE_RUN=$smoke
  fi
  if [ -n "${TRANSFORM:-}" ]; then
    vars=$vars,TRANSFORM=$TRANSFORM,PRL_TRANSFORM_ARGS_FILE=$record/transform_args.json  # JSON has commas
  fi
  elapse=168:00:00
fi
elapse=${ELAPSE:-$elapse}
# nodes: the most any arm of the chain runs on (NNODES; chain.sh start_workers makes the others Ray workers)
nodes=1 arms=()
if has smoke; then arms+=("$SMOKE_ARM"); fi
if has smoke_grpo || has smoke_ppo; then arms+=("${ARM:-ota_train_smoke_transform}"); fi
if has train_eval; then arms+=("$ARM"); fi
for a in "${arms[@]}"; do n=$(arm_setting NNODES "$a"); [ "${n:-1}" -le "$nodes" ] || nodes=$n; done

submit_job() {
  local cmd=("$@")
  if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; return 0; fi
  mkdir -p "$record" "$PRL_EXT/logs"
  printf '%s' "${PRL_TRANSFORM_ARGS:-}" > "$record/transform_args.json"
  # shellcheck disable=SC2046  # one file per transform of the chain
  { date -u +%FT%TZ; printf '%q ' "${cmd[@]}"; echo; [ -n "${TRANSFORM:-}" ] && sha256sum $(printf 'transforms/%s.py ' \
    ${TRANSFORM//+/ }) "$record/transform_args.json"; [ -d transforms/rewrite_prompts ] && sha256sum transforms/rewrite_prompts/*.md; } \
    > "$record/settings.txt"
  "${cmd[@]}" | tee -a "$record/jobs.txt"
}
