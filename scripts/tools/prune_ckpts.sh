#!/usr/bin/env bash
# Keep only the newest *complete* checkpoint (the one latest_checkpointed_iteration.txt names) in each run's ckpt/ (and
# ckpt_critic/, a PPO run's critic);
# older iter_* dirs and empty/partial ones are deleted, except the iterations listed in KEEP_ITERS (space-separated,
# e.g. the checkpoint with the best held-out eval, which is the candidate final model). A PPO run's ckpt_critic/ is
# pruned the same way but keeps only its newest: the critic is needed to resume, never to evaluate.
#   KEEP_ITERS="69" prune_ckpts.sh RUN_DIR [RUN_DIR ...]
# LIVE=1 while a run may be saving into ckpt/: only iterations older than the latest are removed, because a newer
# iter_* dir can be the checkpoint being written (latest_checkpointed_iteration.txt moves only once it is complete).
set -euo pipefail
for r in "$@"; do
  for sub in ckpt ckpt_critic; do
    c=$r/$sub; [ -f "$c/latest_checkpointed_iteration.txt" ] || continue
    keep_iters=${KEEP_ITERS:-}; [ "$sub" = ckpt ] || keep_iters=
    keep=$(printf "iter_%07d" "$(cat "$c/latest_checkpointed_iteration.txt")")
    [ -d "$c/$keep" ] || { echo "$(basename "$r")/$sub: latest $keep missing, nothing pruned"; continue; }
    for d in "$c"/iter_*; do
      b=$(basename "$d"); [ "$b" = "$keep" ] && continue
      [ "${LIVE:-0}" = 1 ] && [[ "$b" > "$keep" ]] && continue   # zero-padded names: string order = iteration order
      for k in $keep_iters; do [ "$b" = "$(printf "iter_%07d" "$k")" ] && continue 2; done
      rm -rf -- "$d"; echo "$(basename "$r")/$sub: removed $b"
    done
  done
done
