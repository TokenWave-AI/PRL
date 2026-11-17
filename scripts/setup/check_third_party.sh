#!/usr/bin/env bash
# Verify the vendored upstream trees are untouched (byte-identical to the pinned upstream commits).
# All our modifications must live in patches/ (applied into build/ by scripts/setup/apply_patches.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
check() {  # name repo_url vendored_dir
  local commit; commit=$(cat "$ROOT/patches/$1/UPSTREAM_COMMIT")
  git clone -q "$2" "$TMP/$1" && git -C "$TMP/$1" checkout -q "$commit"
  if diff -rq -x .git -x __pycache__ -x '*.egg-info' "$TMP/$1" "$3"; then echo "$1: pristine @ ${commit:0:7}"; else echo "$1: MODIFIED (move the change into patches/$1/)"; return 1; fi
}
check miles https://github.com/radixark/miles.git "$ROOT/third_party/miles"
