#!/bin/bash
# Run a subset of tests/fn quickly.
#
# tests/fn holds 36,310 files in a single directory. pytest re-globs
# that directory once PER FILE ARGUMENT, so `pytest tests/fn/a.py
# tests/fn/b.py ...` costs ~13 s per file -- 78 files took over 16
# minutes and looked like a hang. Copying the wanted files into a
# scratch directory (with conftest.py) drops the same 78 files to
# 3.4 seconds. Always use this instead of passing many tests/fn paths.
#
# usage: run_fn_subset.sh <file-with-test-paths> [pytest args...]
set -euo pipefail
LIST="${1:?usage: run_fn_subset.sh <list-file> [pytest args]}"
shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="$(mktemp -d /tmp/fnrun.XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$REPO/tests/fn/conftest.py" "$SCRATCH/"
# shellcheck disable=SC2046
cp $(tr '\n' ' ' < "$LIST") "$SCRATCH/"
cd "$SCRATCH"
exec python -m pytest . -q -p no:randomly -p no:cacheprovider --no-header "$@"
