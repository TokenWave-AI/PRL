#!/bin/bash
# Chain step replay (scripts/lib/chain.sh): train-only checks of the trainer and of the job's failure handling, on the
# recorded batches of an accepted smoke (PRL_REPLAY_SOURCE, its RUN_DIR; Miles --load-debug-rollout-data, no
# inference engine, no sandbox; configs/arms/ota_replay_check.env). One run per case, each in its own directory under
# $PRL_EXT/runs/replay_<job>:
#   base            3 updates from the base model on the smoke's batches 0, 1, 0, saving after updates 1 and 2: its
#                   first two must reproduce the smoke's (the reference curve)
#   resume          base's checkpoint 1 resumed for update 2: the same update as base's (resume consistency)
#   old_checkpoint  the smoke's own checkpoint (older code) resumed for one more update
#   data_parallel   base's first two updates on 4 trainer GPUs instead of 2
#   longest_sample  one update with a sample grown to MAX_CONTEXT (scripts/tools/long_sample_batch.py): memory
#   nan             one update on a batch with a NaN reward: the run stops or the update stays finite, never a NaN
#                   checkpoint
#   kill_save       killed (SIGKILL, as a lost node) while saving checkpoint 1, then resumed: the tracker still names
#                   checkpoint 0 and the resumed run saves 1
#   kill_rank       one trainer rank SIGKILLed during an update: the job ends instead of hanging
#   freeze          one trainer rank SIGSTOPped: the watchdog (scripts/checks/train_watch.py) stops the job
#   sigterm         the job's process group gets SIGTERM, as from the scheduler: nothing of it is left 90 s later
# scripts/checks/replay_check.py verdict judges them into $R/replay.json and fails the step if one failed. Checkpoints
# (117 GB each for the 9B) are deleted when no later case needs them. E.g.
#   CHAIN=replay PRL_REPLAY_SOURCE=$PRL_EXT/runs/grpo_ota_train_smoke_113416 bash scripts/submit.sh
set -uo pipefail
export PRL_ROOT=${PRL_ROOT:-$PWD} PRL_SITE=${PRL_SITE:?set PRL_SITE (configs/sites)}
SRC=${PRL_REPLAY_SOURCE:?the RUN_DIR of an accepted smoke}
R=$PRL_EXT/runs/replay_${JOB_ID:?}
ME=$(id -un)
grep -q '"passed": true' "$SRC/acceptance.json" 2>/dev/null && [ -f "$SRC/traces/rollout_data/1.pt" ] && [ -f "$SRC/ckpt/latest_checkpointed_iteration.txt" ] ||
  { echo "[replay] $SRC is not an accepted smoke with its batches and checkpoint" >&2; exit 1; }
mkdir -p "$R/data"
ln -sfn "$SRC/traces/rollout_data/0.pt" "$R/data/0.pt"
ln -sfn "$SRC/traces/rollout_data/1.pt" "$R/data/1.pt"
ln -sfn "$SRC/traces/rollout_data/0.pt" "$R/data/2.pt"   # a third update needs a third batch
echo "$SRC" > "$R/source"
export DATASET=otagent HARNESS=harbor
set -a; source "$PRL_ROOT/configs/sites/$PRL_SITE.env"; set +a
source "$PRL_ROOT/scripts/lib/env.sh"
if ! command -v apptainer >/dev/null 2>&1; then module load apptainer; fi
source "$PRL_ROOT/scripts/lib/guard.sh"
export RUN_DIR=$R
(set -a; source "$PRL_ROOT/configs/arms/common_ota_single_node.env"; set +a; hardware_gate)   # the arms' GPUs

# start CASE [VAR=VALUE...]: the case's Miles run under the watchdog, in its own session (a drill signals its group);
# an update takes 100-290 s and the first comes within ~10 min, so 30 min without one is a hang
start() {
  local name=$1; shift
  mkdir -p "$R/$name"
  setsid bash -c 'export "$@"; set -a; source "configs/sites/$PRL_SITE.env"; source configs/arms/common_ota_single_node.env
    source configs/arms/ota_replay_check.env; set +a; source scripts/lib/guard.sh; run_guarded "replay"' \
    _ ARM=ota_replay_check PRL_STALL_SECS=1800 PRL_REPLAY_DIR="$R/$name" PRL_REPLAY_DATA="$R/data/{rollout_id}.pt" "$@" \
    >> "$R/$name/case.log" 2>&1 &
  CASE_PID=$!
}
# finish CASE: wait for it, record its exit status and end time, then remove whatever of it is left (a drill's
# orphans: Ray's daemons run in their own sessions)
finish() {
  local rc=0; wait "$CASE_PID" || rc=$?
  echo "$rc" > "$R/$1/exit"; date +%s > "$R/$1/end_time"
  cat "$R/$1/ckpt/latest_checkpointed_iteration.txt" > "$R/$1/tracker" 2>/dev/null || rm -f "$R/$1/tracker"
  pkill -KILL -s "$CASE_PID" 2>/dev/null
  pkill -KILL -u "$ME" -f '^ray::|raylet|gcs_server|ray/dashboard|ray/_private' 2>/dev/null
  sleep 5
  echo "[replay] $1: exit $rc at $(date)"
}
run() { start "$@"; finish "$1"; }
# until FILE PATTERN SECONDS: wait for the pattern in the file (0) or give up (1)
until_seen() {
  local i
  for ((i = 0; i < $3; i += 5)); do grep -q "$2" "$1" 2>/dev/null && return 0; sleep 5; done
  return 1
}
trainer_rank() { pgrep -n -u "$ME" -f '^ray::MegatronTrainRayActor'; }
copy_ckpt() {  # copy_ckpt FROM_CKPT_DIR ITER TO_CKPT_DIR: one checkpoint by hard links, as the latest
  local it; it=$(printf 'iter_%07d' "$2")
  mkdir -p "$3/rollout"
  cp -al "$1/$it" "$3/" && echo "$2" > "$3/latest_checkpointed_iteration.txt"
  cp "$1/rollout/global_dataset_state_dict_$2.pt" "$3/rollout/" 2>/dev/null || true
}
step0='train/step.: 0}'

verdict() { "$PRL_HOST_PYTHON" "$PRL_ROOT/scripts/checks/replay_check.py" verdict "$R"; }
run base PRL_REPLAY_ROLLOUTS=3 PRL_REPLAY_SAVE_INTERVAL=2
# the other cases build on base: without it they would only repeat its failure, each up to the stall limit
[ "$(cat "$R/base/exit")" = 0 ] || { rm -rf "$R/base/ckpt"; verdict; exit 1; }
copy_ckpt "$R/base/ckpt" 1 "$R/resume/ckpt"
rm -rf "$R/base/ckpt/iter_0000002"
run resume PRL_REPLAY_ROLLOUTS=3
rm -rf "$R/resume/ckpt" "$R/base/ckpt"
copy_ckpt "$SRC/ckpt" "$(cat "$SRC/ckpt/latest_checkpointed_iteration.txt")" "$R/old_checkpoint/ckpt"
run old_checkpoint PRL_REPLAY_ROLLOUTS=3
rm -rf "$R/old_checkpoint/ckpt"
run data_parallel PRL_REPLAY_ROLLOUTS=2 PRL_REPLAY_TRAIN_GPUS=4
rm -rf "$R/data_parallel/ckpt"

max_context=$(set -a; source configs/arms/common_ota_single_node.env; source configs/arms/ota_train_smoke.env
  echo "$MAX_CONTEXT")
in_miles python3 "$PRL_ROOT/scripts/tools/long_sample_batch.py" "$R/data/0.pt" "$R/data/long_0.pt" "$max_context" 0 \
  > "$R/data/long.log" 2>&1
run longest_sample PRL_REPLAY_ROLLOUTS=1 PRL_REPLAY_DATA="$R/data/long_{rollout_id}.pt"
rm -rf "$R/longest_sample/ckpt"
in_miles python3 "$PRL_ROOT/scripts/checks/replay_check.py" nan "$R/data/0.pt" "$R/data/nan_0.pt" > "$R/data/nan.log" 2>&1
run nan PRL_REPLAY_ROLLOUTS=1 PRL_REPLAY_DATA="$R/data/nan_{rollout_id}.pt"
rm -rf "$R/nan/ckpt"

# kill_save: SIGKILL everything once checkpoint 1's directory appears (a save takes about a minute)
start kill_save PRL_REPLAY_ROLLOUTS=2 PRL_REPLAY_SAVE_INTERVAL=1
if until_seen "$R/kill_save/train.log" "train/step.: 1}" 3600; then
  # a file test cannot go inside (( )): there "-d" is minus a variable d, unbound under set -u
  for ((i = 0; i < 600; i++)); do [ -d "$R/kill_save/ckpt/iter_0000001" ] && break; sleep 1; done
  sleep 3; date +%s > "$R/kill_save/inject_time"
  kill -KILL -- "-$CASE_PID" 2>/dev/null; pkill -KILL -u "$ME" -f '^ray::|raylet|gcs_server' 2>/dev/null
fi
finish kill_save
cat "$R/kill_save/ckpt/latest_checkpointed_iteration.txt" > "$R/kill_save/tracker_after_kill" 2>/dev/null
ls "$R/kill_save/ckpt" > "$R/kill_save/ckpt_after_kill" 2>/dev/null
mv "$R/kill_save/train.log" "$R/kill_save/train_killed.log" 2>/dev/null
start kill_save PRL_REPLAY_ROLLOUTS=2 PRL_REPLAY_SAVE_INTERVAL=1
finish kill_save
cat "$R/kill_save/ckpt/latest_checkpointed_iteration.txt" > "$R/kill_save/tracker_after_resume" 2>/dev/null
rm -rf "$R/kill_save/ckpt"

# kill_rank, freeze: 30 s into update 1 (an update takes 100-290 s)
start kill_rank PRL_REPLAY_ROLLOUTS=2 PRL_STALL_SECS=1200
if until_seen "$R/kill_rank/train.log" "$step0" 3600; then
  sleep 30; pid=$(trainer_rank); echo "$(date +%s) $pid" > "$R/kill_rank/inject_time"; kill -KILL "$pid"
fi
finish kill_rank
start freeze PRL_REPLAY_ROLLOUTS=2 PRL_STALL_SECS=900  # > model load + update 0
if until_seen "$R/freeze/train.log" "$step0" 3600; then
  sleep 30; pid=$(trainer_rank); echo "$(date +%s) $pid" > "$R/freeze/inject_time"; kill -STOP "$pid"
fi
finish freeze
start sigterm PRL_REPLAY_ROLLOUTS=2
if until_seen "$R/sigterm/train.log" "$step0" 3600; then
  sleep 30; date +%s > "$R/sigterm/inject_time"; kill -TERM -- "-$CASE_PID"
  sleep 90; pgrep -u "$ME" -fa '^ray::|raylet|gcs_server|apptainer|run_job.sh|train.py' > "$R/sigterm/left_after_90s" || true
fi
finish sigterm
rm -rf "$R"/*/ckpt
verdict
