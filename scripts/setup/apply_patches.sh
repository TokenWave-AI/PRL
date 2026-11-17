#!/usr/bin/env bash
# Build the patched copy of the vendored upstream tree:
#   third_party/miles/ + patches/miles/*.patch -> build/miles/
# The vendored dirs stay byte-identical to upstream; everything we change lives in patches/.
# Idempotent: rebuilds only when the patches or the upstream tree changed (stamp file).
# Each build lives in its own dir build/<name>@<stamp>; build/<name> is a symlink to the newest one, and
# launchers resolve it once (PRL_MILES_DIR in scripts/lib/env.sh). A rebuild therefore never touches a
# tree that a running job uses (swapping it in place removed the head's cwd and changed code under running jobs).
# Runtime code (PYTHONPATH, train.py) uses build/, never the vendored dirs directly.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/build"
exec 9>"$ROOT/build/.lock"; flock 9   # concurrent launchers (e.g. several slurm jobs) build once

build_one() {  # name src_dir
  local name=$1 src=$2 pdir="$ROOT/patches/$1" out="$ROOT/build/$1"
  local stamp
  stamp=$( { cat "$pdir/UPSTREAM_COMMIT"; cat "$pdir"/*.patch; \
             find "$src" -type f -not -path '*/__pycache__/*' -not -path '*.egg-info/*' -printf '%P %s %T@\n' | sort; } | sha256sum | cut -c1-16)
  local ver="$out@$stamp"
  if [ -L "$out" ] && [ "$(readlink "$out")" = "$name@$stamp" ] && [ -f "$ver/.patch_stamp" ]; then
    echo "[apply_patches] $name: up to date ($stamp)"; return
  fi
  if [ ! -f "$ver/.patch_stamp" ]; then
    local tmp="$ROOT/build/.$name.tmp.$$"
    rm -rf "$tmp"; mkdir -p "$tmp"
    rsync -a --exclude __pycache__ --exclude '*.egg-info' "$src/" "$tmp/"
    for p in "$pdir"/*.patch; do
      # GIT_CEILING_DIRECTORIES: build/ sits inside our own git checkout; without it `git apply` resolves
      # paths against the outer repo and silently skips every hunk (while exiting 0).
      (cd "$tmp" && export GIT_CEILING_DIRECTORIES="$ROOT/build" && git apply --check -p1 "$p" && git apply -p1 "$p" \
         && git apply -R --check -p1 "$p") \
        || { echo "[apply_patches] $name: FAILED to apply $(basename "$p")" >&2; rm -rf "$tmp"; exit 1; }
      echo "[apply_patches] $name: applied $(basename "$p")"
    done
    echo "$stamp" > "$tmp/.patch_stamp"
    rm -rf "$ver"; mv "$tmp" "$ver"
  fi
  # Legacy layout: build/<name> was a real directory. Rename it (running jobs keep their open cwd/inodes).
  if [ -d "$out" ] && [ ! -L "$out" ]; then mv "$out" "$out@legacy-$(cat "$out/.patch_stamp" 2>/dev/null || echo x)"; fi
  ln -sfn "$name@$stamp" "$ROOT/build/.$name.lnk.$$" && mv -T "$ROOT/build/.$name.lnk.$$" "$out"
  echo "[apply_patches] $name: built $ver -> $out"
}

build_one miles "$ROOT/third_party/miles"
