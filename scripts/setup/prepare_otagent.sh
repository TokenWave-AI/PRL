#!/bin/bash
# One-off per site: OpenThoughts-Agent-v1-RL (728 Harbor tasks) + OpenThoughts-TB-dev (70) as task dirs + Miles JSONL,
# and the Qwen3.5 weights, for the Harbor/Terminus-2 harness (HARNESS=harbor, configs/datasets/otagent.env).
# Idempotent; needs internet (HF). Paths from configs/sites/<site>.env (PRL_EXT, HF_HOME):
#   a compute node of the site (not a login node), from the repo root:
#              PRL_SITE=<site> MODELS=Qwen/Qwen3.5-9B bash scripts/setup/prepare_otagent.sh
#   a machine without the Miles image (e.g. the one that builds the task images): PRL_EXT=<dir> HF_HOME=<dir> \
#              PRL_SCRATCH=<dir> PREP_PYTHON=<venv python> HF_CLI=<venv hf> MODELS= bash scripts/setup/prepare_otagent.sh
# Layout: $PRL_EXT/data/{otagent_raw/tasks.parquet, ot_tbdev_raw/<uuid>/, harbor_tasks/<task_* + uuid>/,
#         otagent_rl/train.jsonl, ot_tbdev/tbdev.jsonl}. harbor_tasks must hold REAL dirs (Miles resolves symlinks).
set -euo pipefail
# Under qsub the script runs from a spool copy, so $0 is useless: submit from the repo root (-cwd) or set PRL_ROOT.
export PRL_ROOT=${PRL_ROOT:-$PWD}; [ -f "$PRL_ROOT/scripts/lib/env.sh" ] || PRL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PRL_ROOT"; source scripts/lib/env.sh
mkdir -p "$PRL_SCRATCH"
MODELS=${MODELS-Qwen/Qwen3.5-9B}            # space-separated HF ids; MODELS= skips
D=$PRL_EXT/data; mkdir -p "$D"
# Python for hf download / parquet extraction: the Miles container (has hf, pyarrow) when its image exists (a SIF, or
# the directory the chain steps unpack it to), else PREP_PYTHON.
if [ -e "$MILES_IMAGE" ]; then run_py() { in_miles bash -c "cd $PRL_ROOT && $*"; }; HF=hf
else run_py() { bash -c "cd $PRL_ROOT && $*"; }; HF=${HF_CLI:-hf}; fi   # e.g. HF_CLI=<venv>/bin/hf
PY=${PREP_PYTHON:-python3}; [ -e "$MILES_IMAGE" ] && PY=python3
[ -z "$MODELS" ] || echo "== 1. weights -> $HF_HOME/hub"
for m in $MODELS; do
  if ls -d "$HF_HOME/hub/models--${m//\//--}"/snapshots/*/config.json >/dev/null 2>&1; then echo "   $m present"; else
    run_py "$HF download $m >/dev/null && echo '   $m downloaded'"; fi
done
echo "== 2. raw downloads"
[ -f "$D/otagent_raw/tasks.parquet" ] || run_py "$HF download open-thoughts/OpenThoughts-Agent-v1-RL --repo-type dataset --local-dir $D/otagent_raw >/dev/null"
[ "$(ls -d "$D"/ot_tbdev_raw/*/task.toml 2>/dev/null | wc -l)" -ge 70 ] || run_py "$HF download open-thoughts/OpenThoughts-TB-dev --repo-type dataset --local-dir $D/ot_tbdev_raw >/dev/null"
echo "   parquet: $(du -h "$D/otagent_raw/tasks.parquet" | cut -f1); TB-dev task dirs: $(ls -d "$D"/ot_tbdev_raw/*/task.toml | wc -l)"
echo "== 3. task dirs -> $D/harbor_tasks (train task_* from the parquet; TB-dev copied, ids disjoint)"
mkdir -p "$D/harbor_tasks"
run_py "$PY - <<'PYEOF'
import io, os, sys, tarfile
from pathlib import Path, PurePosixPath
import pyarrow.parquet as pq
D = Path('$D'); out = D / 'harbor_tasks'
t = pq.read_table(D / 'otagent_raw' / 'tasks.parquet')
n = 0
for tid, blob in zip(t.column('path').to_pylist(), t.column('task_binary').to_pylist()):
    dest = out / Path(str(tid)).name
    if (dest / 'task.toml').is_file(): continue
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bytes(blob)), mode='r:*') as tf:
        for m in tf.getmembers():
            parts = [p for p in PurePosixPath(m.name).parts if p not in ('..', '.', '', '/')]
            if not parts or '.snapshot' in parts: continue
            tgt = dest.joinpath(*parts)
            if os.path.commonpath([dest.resolve(), tgt.resolve()]) != str(dest.resolve()): raise RuntimeError(m.name)
            if m.isdir(): tgt.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                tgt.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as s, open(tgt, 'wb') as d: d.write(s.read())
                if parts[-1].endswith('.sh'): tgt.chmod(0o755)
    n += 1
print(f'   extracted {n} new train task dirs')
PYEOF"
for t in "$D"/ot_tbdev_raw/*/; do
  id=$(basename "$t"); [ -f "$t/task.toml" ] || continue
  [ -f "$D/harbor_tasks/$id/task.toml" ] || { rm -rf "$D/harbor_tasks/$id"; cp -r "$t" "$D/harbor_tasks/$id"; rm -rf "$D/harbor_tasks/$id"/environment/__pycache__; }
done
echo "   harbor_tasks: $(ls -d "$D"/harbor_tasks/task_* | wc -l) train + $(ls "$D/harbor_tasks" | grep -vc '^task_') TB-dev dirs"
echo "== 4. JSONL (prl.datasets.harbor.convert)"
TMP=$(mktemp -d "${PRL_SCRATCH:-/tmp}/otprep.XXXX"); mkdir -p "$TMP/train" "$TMP/tbdev"
for d in "$D"/harbor_tasks/task_*; do ln -s "$d" "$TMP/train/"; done
for d in "$D"/harbor_tasks/*/; do case "$(basename "$d")" in task_*) ;; *) ln -s "${d%/}" "$TMP/tbdev/";; esac; done
run_py "$PY -m prl.datasets.harbor.convert --tasks-dir $TMP/train --out $D/otagent_rl/train.jsonl --data-source open-thoughts/OpenThoughts-Agent-v1-RL && \
        $PY -m prl.datasets.harbor.convert --tasks-dir $TMP/tbdev --out $D/ot_tbdev/tbdev.jsonl --data-source open-thoughts/OpenThoughts-TB-dev"
rm -rf "$TMP"
echo "== prepare_otagent: ALL OK"; wc -l "$D"/otagent_rl/train.jsonl "$D"/ot_tbdev/tbdev.jsonl
