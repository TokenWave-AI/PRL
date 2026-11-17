#!/bin/bash
# Pre-run check of the trajectory transform (prl/transforms/check.py) in the Miles container, as the trainer will
# run it (same image, binds and PYTHONPATH as scripts/lib/env.sh in_miles): a transform that fails on the recorded
# trials stops the job before training. No transform selected: returns at once.
#   source scripts/lib/env.sh; source scripts/lib/transform_check.sh; transform_check || exit 1
# Needs RUN_DIR and MODEL_HF_DIR (the tokenizer); passes the transform's, the judge's and EXTRA_ARGS (--use-tis) env.
transform_check() {
  if ! command -v apptainer >/dev/null 2>&1; then module load apptainer; fi
  local pass=() v
  for v in RUN_DIR MODEL_HF_DIR EXTRA_ARGS $(compgen -e | grep -E '^(PRL_TRANSFORM|PRL_JUDGE_)'); do
    pass+=("$v=${!v}")
  done
  mkdir -p "$PRL_SCRATCH/home"
  in_miles env "${pass[@]}" /opt/sglang/bin/python3 -m prl.transforms.check
}
