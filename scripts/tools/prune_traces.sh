#!/usr/bin/env bash
# Prune Miles' --dump-details tensor dumps, which grow ~0.4 GB per step: rollout_data/<id>.pt, train_data/<id>_<rank>.pt
# and policy_loss_debug/rank_*_call_*.pt (one file per micro-batch call). Keeps the newest KEEP rollout ids
# (and the newest KEEP*CALLS policy_loss_debug files) for debugging. Trajectories (traces/trajectory/*.jsonl),
# dashboard columns and events are kept: they are small and are what the analysis tools read.
#   prune_traces.sh RUN_DIR [KEEP=3]
set -euo pipefail
T=${1:?run dir}/traces; KEEP=${2:-3}
[ -d "$T" ] || exit 0
before=$(du -sm "$T" | cut -f1)
prune_ids() {  # dir pattern-to-id-sed
  local d=$1 sedexpr=$2
  [ -d "$d" ] || return 0
  local keep_ids
  keep_ids=$(ls "$d" | sed -E "$sedexpr" | sort -n -u | tail -n "$KEEP")
  [ -n "$keep_ids" ] || keep_ids="__none__"   # KEEP=0: an empty list would match the empty string
  for f in "$d"/*; do
    id=$(basename "$f" | sed -E "$sedexpr")
    grep -qx "$id" <<< "$keep_ids" || rm -f -- "$f"
  done
}
if [ -d "$T/policy_loss_debug" ]; then   # call index is global: keep the newest files by mtime
  # files per step estimated from the number of steps so far (trajectory/<id>.jsonl is never pruned)
  n=$(ls "$T/policy_loss_debug" | wc -l); per=$(( n / ($(ls "$T/trajectory" 2>/dev/null | wc -l) + 1) + 1 ))
  ls -t "$T/policy_loss_debug" | tail -n +$(( KEEP * per + 1 )) | sed "s|^|$T/policy_loss_debug/|" | xargs -r rm -f --
fi
prune_ids "$T/rollout_data" 's/^([0-9]+)\.pt$/\1/'
prune_ids "$T/train_data" 's/^([0-9]+)_[0-9]+\.pt$/\1/'
echo "$(basename "$1"): traces ${before} MB -> $(du -sm "$T" | cut -f1) MB"
