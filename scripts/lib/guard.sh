# Shared by the chain steps that train (scripts/chain/grpo_smoke.sh, grpo_full.sh). Source me.

# hardware_gate: this node has the arm's NUM_GPUS GPUs, each with the arm's PRL_GPU_MIN_MIB of memory, of the
# site's PRL_GPU_NAME if it sets one (e.g. H100, GB200), and with no uncorrected ECC error or pending / failed row
# remap (fields a GPU does not report, [N/A], are skipped); RUN_DIR and PRL_SCRATCH take a 1 GiB write, read back and
# delete (a full quota fails here instead of mid-run). The GPU list, health fields and df / df -i are kept in
# $RUN_DIR/hardware.csv and node_health.txt.
hardware_gate() {
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$RUN_DIR/hardware.csv"
  { nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total,remapped_rows.pending,remapped_rows.failure \
      --format=csv,noheader 2>&1 || echo "health query unavailable"
    df -h "$RUN_DIR" "$PRL_SCRATCH"; df -i "$RUN_DIR" "$PRL_SCRATCH"; } > "$RUN_DIR/node_health.txt"
  local d
  for d in "$RUN_DIR" "$PRL_SCRATCH"; do
    dd if=/dev/zero of="$d/.write_check_$$" bs=1M count=1024 status=none && cmp -s -n 1073741824 "$d/.write_check_$$" /dev/zero ||
      { rm -f "$d/.write_check_$$"; echo "Storage gate: cannot write and read back 1 GiB in $d" >&2; return 1; }
    rm -f "$d/.write_check_$$"
  done
  "$PRL_HOST_PYTHON" - <<'PY'
import os
from pathlib import Path
run = Path(os.environ['RUN_DIR'])
gpus = (run / 'hardware.csv').read_text().splitlines()
name = os.environ.get('PRL_GPU_NAME', '')
want, min_mib = int(os.environ.get('NUM_GPUS') or 1), int(os.environ.get('PRL_GPU_MIN_MIB') or 0)
assert len(gpus) >= want and all(name in gpu and int(gpu.split(',')[1].split()[0]) >= min_mib for gpu in gpus), (
    f'need {want} GPUs ({name or "any model"}, >= {min_mib} MiB each), found: {gpus}')
health = [line.split(', ') for line in (run / 'node_health.txt').read_text().splitlines()[:len(gpus)]]
bad = [row for row in health if len(row) == 4 and any(v.strip() == 'Yes' or v.strip().isdigit() and v.strip() != '0'
                                                       for v in row[1:])]
assert not bad, f'GPU health: uncorrected ECC errors or row remaps (index, ecc, pending, failure): {bad}'
print(f'Hardware gate: {len(gpus)} {name or "GPUs"} (need {want}), each >= {min_mib} MiB, healthy; storage writable',
      flush=True)
PY
}

# run_guarded WHAT [COMMAND]: the arm's Miles run (scripts/run_job.sh with COMMON=common_ota_single_node, ARM) under
# the watchdog scripts/checks/train_watch.py (fatal call-tag line, no update for too long, mass discards: it reads only
# what this job appends to train.log), running COMMAND (e.g. the full run's checkpoint pruning) once a minute meanwhile.
# Returns the run's exit status.
run_guarded() {
  local what=$1 every=${2:-} offset
  export COMMON=common_ota_single_node ARM
  offset=$(stat -c %s "$RUN_DIR/train.log" 2>/dev/null || echo 0)
  bash "$PRL_ROOT/scripts/run_job.sh" &
  local runner=$!
  "$PRL_HOST_PYTHON" "$PRL_ROOT/scripts/checks/train_watch.py" "$RUN_DIR/train.log" "$offset" "$runner" "$what" &
  local watch=$!
  ( while kill -0 "$runner" 2>/dev/null; do
      [ -n "$every" ] && { eval "$every" || true; }
      sleep 60
    done ) &
  local chores=$!
  trap "kill $watch $chores $runner 2>/dev/null || true" EXIT
  wait "$runner"
}
