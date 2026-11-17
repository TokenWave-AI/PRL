# Pack a finished job's Harbor trial directories into one tar each (a 68-update run writes 200-300k small files
# under harbor_trials/, and a group quota of 27.9M files ran out). Sourced by
# scripts/lib/chain.sh, which packs the job's own trial directories when the chain ends; usable by hand on older runs:
#   source scripts/lib/archive_trials.sh; archive_trials <run>/harbor_trials/job_123 [...]
# A directory becomes <dir>.tar next to it (written as <dir>.tar.part first) and is removed only after the tar lists
# exactly as many files as the directory holds; on any mismatch or failure the directory stays and the .part is
# deleted. Nothing reads a finished job's trial directories (analyses use the run's traces/ dumps; the transform
# pre-check uses tests/fixtures). PRL_ARCHIVE_TRIALS=0 turns it off (default 1).

archive_trials() {
  [ "${PRL_ARCHIVE_TRIALS:-1}" = 1 ] || { echo "[archive] off (PRL_ARCHIVE_TRIALS=${PRL_ARCHIVE_TRIALS})"; return 0; }
  local d parent base tarf n_fs n_tar
  for d in "$@"; do
    [ -d "$d" ] || continue
    n_fs=$(find "$d" -type f | wc -l)
    [ "$n_fs" -gt 0 ] || continue
    parent=$(dirname "$d"); base=$(basename "$d"); tarf="$parent/$base.tar"
    if [ -e "$tarf" ]; then echo "[archive] $tarf exists; $d left as it is"; continue; fi
    if ! tar -cf "$tarf.part" -C "$parent" "$base"; then
      echo "[archive] tar failed for $d; directory kept"; rm -f "$tarf.part"; continue
    fi
    n_tar=$(tar -tf "$tarf.part" | grep -vc '/$')
    if [ "$n_tar" != "$n_fs" ]; then
      echo "[archive] $d: $n_fs files but the tar lists $n_tar; directory kept"; rm -f "$tarf.part"; continue
    fi
    mv "$tarf.part" "$tarf"
    rm -rf "$d" || true  # Lustre can refuse an emptied directory for a moment: the files are what count
    echo "[archive] $d: $n_fs files -> $(basename "$tarf") ($(du -h "$tarf" | cut -f1)); $(find "$d" -type f 2>/dev/null | wc -l) files left"
  done
}
