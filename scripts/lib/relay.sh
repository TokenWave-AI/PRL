# Relay one training run across consecutive jobs (a run longer than one job's time limit; also a job resubmitted after
# a failure). Sourced by
# scripts/chain/grpo_full.sh. Every job of a chain uses the same RUN_DIR, and Miles resumes by itself:
# --load = --save = $RUN_DIR/ckpt restores the weights, the Adam state and the dataset cursor of the iteration in
# ckpt/latest_checkpointed_iteration.txt, starts at the next rollout_id and syncs the loaded weights to SGLang before
# the first rollout (third_party/miles train.py, miles/backends/megatron_utils/model.py). These helpers only decide
# whether a job should train, and record each start in $RUN_DIR/relay.tsv (job ID, start iteration, UTC time).
# Configuration: PRL_RELAY_RESET=1 accepts a start without progress since the previous job (a resubmission
# after a fix); PRL_RESUME_DATA_FROM_PREVIOUS=1 resumes a checkpoint that has no dataset cursor (below).

latest_iter() {   # last complete checkpoint iteration (= Miles rollout_id), -1 if none; $2 = ckpt subdir
  local f=$1/${2:-ckpt}/latest_checkpointed_iteration.txt
  if [ -s "$f" ]; then tr -d '[:space:]' < "$f"; else echo -1; fi
}

# critic_load RUN_DIR BASE -> the PPO critic's --critic-load. Miles defaults it to --load, which on a resume is
# the actor's own checkpoint dir (upstream test_an_existing_checkpoint_is_left_alone); the critic saves to
# RUN_DIR/ckpt_critic. No actor checkpoint yet (fresh run, or still inside --num-critic-only-steps, which save only
# the critic): BASE, since the actor restarts from the reference weights and Miles requires both to restore the
# same rollout. Otherwise the critic must have saved that same iteration.
critic_load() {
  local run=$1 base=$2 actor critic
  actor=$(latest_iter "$run") critic=$(latest_iter "$run" ckpt_critic)
  if [ "$actor" = -1 ]; then echo "$base"; return; fi
  [ "$critic" = "$actor" ] || { echo "[relay] actor checkpoint $actor but critic $critic in $run" >&2; return 1; }
  echo "$run/ckpt_critic"
}

# relay_begin RUN_DIR NUM_ROLLOUT -> 0 train | 3 run already complete | 1 no checkpoint since the previous job
relay_begin() {
  local run=$1 total=$2 latest prev
  latest=$(latest_iter "$run")
  if [ "$latest" -ge $((total - 1)) ]; then
    echo "[relay] $run complete at iteration $latest of $((total - 1)); nothing to do"
    return 3
  fi
  mkdir -p "$run"
  # Miles saves the weights (and moves the tracker) before the dataset cursor (train.py save()): a job killed in
  # between leaves iteration N without ckpt/rollout/global_dataset_state_dict_N.pt, and Miles would only warn and
  # restart the data order from the first prompt. Stop instead, before recording the start (a refused job is no
  # attempt); PRL_RESUME_DATA_FROM_PREVIOUS=1 uses N-1's cursor (update N's prompts are drawn again once).
  local data=$run/ckpt/rollout/global_dataset_state_dict_$latest.pt
  if [ "$latest" -ge 0 ] && [ ! -f "$data" ]; then
    if [ "${PRL_RESUME_DATA_FROM_PREVIOUS:-0}" = 1 ] && [ -f "${data%_$latest.pt}_$((latest - 1)).pt" ]; then
      cp "${data%_$latest.pt}_$((latest - 1)).pt" "$data"
      echo "[relay] no dataset cursor for iteration $latest: using iteration $((latest - 1))'s" >&2
    else
      echo "[relay] checkpoint $latest has no dataset cursor ($data); PRL_RESUME_DATA_FROM_PREVIOUS=1 resumes" \
        "with iteration $((latest - 1))'s" >&2
      return 1
    fi
  fi
  prev=$(tail -n 1 "$run/relay.tsv" 2>/dev/null | cut -f 2 || true)   # empty for the first job
  printf '%s\t%s\t%s\n' "${JOB_ID:-local}" "$latest" "$(date -u +%FT%TZ)" >> "$run/relay.tsv"
  # A successor job may start whatever its predecessor's exit status. A chain that stopped advancing has a
  # deterministic failure: stop it here instead of repeating it in every remaining job.
  if [ "$prev" = "$latest" ] && [ "${PRL_RELAY_RESET:-0}" != 1 ]; then
    echo "[relay] no new checkpoint since the previous job (still iteration $latest); stopping the chain" >&2
    return 1
  fi
  echo "[relay] $run: starting after iteration $latest"
}

# relay_claim RUN_DIR: one job at a time per run. Two chains with the same PRL_RUN_ID were queued together
# twice; running together they would write and prune the same ckpt/.
# The job keeps $RUN_DIR/active_job (its JOB_ID) fresh every 10 s while this shell lives and removes it after; another
# job finding it fresh (< 2 min) waits up to 3 min (a predecessor still exiting), then refuses (return 1). A job killed
# outright leaves a stale file, which its successor takes over.
relay_claim() {
  local run=$1 f=$1/active_job me=${JOB_ID:-local-$$} parent=$$ i
  mkdir -p "$run"
  for i in $(seq 18); do
    [ -f "$f" ] && [ "$(cat "$f" 2>/dev/null)" != "$me" ] && [ $(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0) )) -lt 120 ] ||
      break
    [ "$i" = 18 ] && { echo "[relay] $run is in use by job $(cat "$f") ($f); not starting" >&2; return 1; }
    sleep 10
  done
  echo "$me" > "$f"
  ( while kill -0 "$parent" 2>/dev/null; do touch "$f"; sleep 10; done
    [ "$(cat "$f" 2>/dev/null)" = "$me" ] && rm -f "$f" ) > /dev/null 2>&1 &
}
