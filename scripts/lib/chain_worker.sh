#!/bin/bash
# A non-head node of a multi-node chain job (e.g. a MoE arm on 2 nodes), started from the head by
# start_workers (scripts/lib/chain.sh) through the site's remote start ($PRL_REMOTE_EXEC, e.g. pjrsh under PJM):
#   chain_worker.sh prepare RUN_DIR       unpack the image on this node's NVMe (as the head's sandbox step); returns
#   chain_worker.sh join RUN_DIR HEAD_IP  join the head's Ray cluster from inside the Miles container until it stops
# The environment is the head's arm as resolved for the job ($RUN_DIR/worker.env, written by start_workers): Miles
# places trainer ranks and SGLang engines here, and they read the same settings (e.g. PRL_FUSED_LOGPROB_CHUNK).
# Logs: $RUN_DIR/worker_<host>.log.
set -uo pipefail
MODE=$1 RUN_DIR=$2 HEAD_IP=${3:-}
exec >> "$RUN_DIR/worker_$(hostname -s).log" 2>&1
set -x
set -a; source "$RUN_DIR/worker.env"; set +a
cd "$PRL_ROOT" || exit 1
# the site's defaults (PATH to its apptainer wrapper, image), then the head's values over them. pjrsh passes the head's
# environment too, whose MILES_IMAGE may already be the head's unpacked image on its own NVMe (a chain whose earlier
# step ran sandbox, e.g. download+convert+smoke, then fails with "no such file"): start from the site's image
unset MILES_IMAGE PRL_SIF
set -a; source "configs/sites/$PRL_SITE.env"; source "$RUN_DIR/worker.env"; set +a
export PRL_SKIP_APPLY_PATCHES=1   # the head built build/ already
source scripts/lib/chain.sh      # sandbox
case $MODE in
prepare)
  echo "worker $(hostname -s) prepare at $(date)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  sandbox || exit 1
  echo "worker $(hostname -s) ready at $(date)" ;;
join)
  sandbox || exit 1   # unpacked by prepare: only sets MILES_IMAGE
  source scripts/lib/env.sh
  mkdir -p "$PRL_SCRATCH/ray_tmp" ${PRL_RAY_PLASMA_DIR:+"$PRL_RAY_PLASMA_DIR"}
  ip=$(/usr/sbin/ip -4 -o addr show "${PRL_NODE_IFACE:?}" | awk '{print $4}' | cut -d/ -f1)
  # As the head (scripts/lib/container_main.sh): the object store on NVMe (/dev/shm under PJM), no Ray memory
  # monitor, no uvloop, in-process inductor compiles, NCCL and gloo sockets on the inter-node interface
  store=""
  [ -n "${PRL_RAY_PLASMA_DIR:-}" ] && store="--plasma-directory=$PRL_RAY_PLASMA_DIR --object-store-memory=${PRL_RAY_OBJECT_STORE:-100000000000}"
  # The head starts Ray only after its preflight: retry until it is up; a join that held for minutes ended with the job
  deadline=$(( $(date +%s) + ${PRL_WORKER_JOIN_S:-3600} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    start=$(date +%s)
    in_miles bash -c "export RAY_TMPDIR='$PRL_SCRATCH/ray_tmp' RAY_memory_monitor_refresh_ms=0 RAY_USE_UVLOOP=0 \
      MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
      RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 no_proxy=127.0.0.1,localhost,$HEAD_IP \
      NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME:-$PRL_NODE_IFACE}' GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME:-$PRL_NODE_IFACE}'; \
      ray start --address='$HEAD_IP:6379' --node-ip-address '$ip' --num-gpus '${NUM_GPUS:?}' --disable-usage-stats $store --block"
    [ $(( $(date +%s) - start )) -gt 300 ] && break
    sleep 15
  done
  echo "worker $(hostname -s) left the cluster at $(date)" ;;
*) echo "chain_worker.sh prepare|join RUN_DIR [HEAD_IP]" >&2; exit 2 ;;
esac
