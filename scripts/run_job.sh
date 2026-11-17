#!/bin/bash
# One Miles run in the container on this node, from the repo root: the site profile, then the arm, then
# scripts/lib/env.sh, then scripts/run.sh. Called by the chain steps (scripts/chain/*.sh, scripts/lib/chain.sh) inside a
# batch job (scripts/pjm/job.sh, scripts/pbs/job.sh); not submitted by itself:
#   PRL_SITE=<site> COMMON=common_ota_single_node ARM=<arm> bash scripts/run_job.sh
set -euo pipefail
export PRL_SITE=${PRL_SITE:?set PRL_SITE (configs/sites)} PRL_ROOT=${PRL_ROOT:-$PWD}
# Order matters: the arm files may set DATASET (and read $PRL_EXT/$PRL_SCRATCH), and env.sh resolves the DATASET preset
# (PROMPT_DATA, HARNESS, ...) when it is sourced. So: site paths -> arms -> env.sh.
set -a; source "$PRL_ROOT/configs/sites/$PRL_SITE.env"; set +a
[ -n "${COMMON:-}" ] && { set -a; source "$PRL_ROOT/configs/arms/$COMMON.env"; set +a; }
[ -n "${ARM:-}" ] && { set -a; source "$PRL_ROOT/configs/arms/$ARM.env"; set +a; export RUN_DIR=${RUN_DIR:-$PRL_EXT/runs/grpo_$ARM}; }
source "$PRL_ROOT/scripts/lib/env.sh"     # DATASET preset + MILES_IMAGE (+ the site profile again, idempotent)
if [ "${HARNESS:-}" = harbor ] && [ "${PRL_SANDBOX_PROVIDER:-local}" = e2b ]; then
  : "${E2B_API_URL:?set the sandbox endpoint in the site profile (configs/sites/example.env)}"
  : "${E2B_DOMAIN:?set the sandbox domain in the site profile}"
  : "${E2B_API_KEY_FILE:?set the sandbox key file in the site profile}"
  [[ "$E2B_API_URL" = https://* ]] && [ -s "$E2B_API_KEY_FILE" ] ||
    { echo "[run_job] E2B_API_URL must be https://... and E2B_API_KEY_FILE a non-empty file" >&2; exit 1; }
  unset E2B_API_KEY E2B_SANDBOX_URL
elif [ "${HARNESS:-}" = harbor ]; then
  # every Harbor arm sets PRL_SANDBOX_PROVIDER=e2b (configs/arms/common_ota.env)
  echo "[run_job] HARNESS=harbor needs PRL_SANDBOX_PROVIDER=e2b" >&2; exit 1
fi
echo "[run_job] job ${JOB_ID:-?} on $(hostname) | DATASET=$DATASET MODE=${MODE:-rollout} MODEL=$MODEL ($MODEL_HF_DIR)"
echo "[run_job] PRL_EXT=$PRL_EXT PRL_SCRATCH=$PRL_SCRATCH MILES_IMAGE=$MILES_IMAGE"
nvidia-smi -L
# A SIF, or the image unpacked to a directory (the chain step sandbox, scripts/lib/chain.sh)
[ -e "$MILES_IMAGE" ] || { echo "Miles image missing: $MILES_IMAGE (pull it once per site: docs/setup.md, step 2)" >&2; exit 1; }
mkdir -p "$PRL_SCRATCH" "$PRL_EXT/logs"
# Snapshot the launcher scripts: bash reads scripts lazily, so editing them mid-run breaks the job.
SNAP=$PRL_SCRATCH/launcher_${JOB_ID:-$$}; mkdir -p "$SNAP/lib" && cp scripts/run.sh "$SNAP/" && cp scripts/lib/*.sh "$SNAP/lib/"
if [ -n "${RUN_DIR:-}" ]; then   # arm runs: keep the configs used, and a host-memory log (the trainer's CPU Adam state + the
  mkdir -p "$RUN_DIR"              # RolloutExecutor + SGLang share one node's RAM; an OOM kill leaves no other trace)
  for a in ${COMMON:+configs/arms/$COMMON.env} ${ARM:+configs/arms/$ARM.env} "configs/models/$MODEL.env"; do cp "$a" "$RUN_DIR/"; done
  # What this job ran, kept per job (relay jobs share RUN_DIR): every code file's sha256, all configs, the resolved
  # environment without secrets (keys live in files; their *_FILE paths stay), the Miles build and image, the GPUs;
  # container_main.sh adds the container's pip freeze (PRL_PROVENANCE_DIR)
  export PRL_PROVENANCE_DIR=$RUN_DIR/provenance/job_${JOB_ID:-local-$(date +%s)}
  mkdir -p "$PRL_PROVENANCE_DIR"
  ( find prl transforms scripts configs patches tests -type f ! -path '*/__pycache__/*' -print0 | sort -z |
      xargs -0 sha256sum > "$PRL_PROVENANCE_DIR/code.sha256"
    tar -czf "$PRL_PROVENANCE_DIR/configs.tar.gz" configs
    env | sort | awk -F= '$1 ~ /_FILE$/ || $1 !~ /(KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|TOKEN|AUTH)/ || $1 ~ /^PRL_/' \
      | grep -vE '^PRL_[A-Z_]*(KEY|SECRET|PASSWORD)=' > "$PRL_PROVENANCE_DIR/env.txt"
    { echo "host $(hostname) job ${JOB_ID:-?} $(date -u +%FT%TZ)"; echo "PRL_MILES_DIR $(readlink -f "${PRL_MILES_DIR:-build/miles}")"
      echo "MILES_IMAGE $MILES_IMAGE $(stat -c '%s bytes, modified %y' "$MILES_IMAGE")"
      nvidia-smi --query-gpu=index,name,serial,uuid,driver_version,vbios_version,memory.total --format=csv; } \
      > "$PRL_PROVENANCE_DIR/system.txt" ) || echo "[run_job] provenance incomplete" >&2
  echo "job ${JOB_ID:-?} start $(date) on $(hostname)" >> "$RUN_DIR/jobs.txt"
  CG=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
  ( while true; do
      echo "MEM $(date +%T) cgroup_current=$(awk '{printf "%.1fG",$1/1073741824}' /sys/fs/cgroup$CG/memory.current 2>/dev/null) max=$(awk '{printf "%.1fG",$1/1073741824}' /sys/fs/cgroup$CG/memory.max 2>/dev/null) free=$(free -g | awk '/Mem/{print $7"G avail"}')"
      ps -eo rss,pid,comm --sort=-rss | head -6 | awk 'NR>1{printf "   %.1fG %s %s\n",$1/1048576,$2,$3}'
      sleep 60; done >> "$RUN_DIR/host_mem_${JOB_ID:-local}.log" 2>&1 ) &
  MEMLOG=$!; trap 'kill $MEMLOG 2>/dev/null' EXIT
fi
bash "$SNAP/run.sh"   # its defaults (MODE, ROLLOUT_BATCH, ...) apply to what the arm left unset
