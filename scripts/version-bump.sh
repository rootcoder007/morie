#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# version-bump.sh — propagate a new MORIE version into all tracked
# CURRENT (and explicitly-whitelisted legacy) locations.
#
# DRY contract:
#   1. Update the canonical VERSION file.
#   2. Walk VERSION_INVENTORY.csv and patch:
#        - every row whose Status == CURRENT
#        - every row whose Version matches the KNOWN_LEGACY_VERSIONS list
#      (the second case catches stale stamps that were left behind in a
#      previous release; the whitelist prevents accidentally rewriting
#      external dependency versions such as `DoubleML>=0.7.1` or
#      `python>=3.7`).
#   3. Re-run scripts/version-inventory.sh to refresh the CSV.
#
# Usage:
#     scripts/version-bump.sh 0.9.6.0
#
# After this script finishes you should commit:
#     VERSION
#     VERSION_INVENTORY.csv
#     <every file the script reports as patched>
#
# Two preserved quirks-of-bash fixes vs. the original draft:
#   - `((COUNTER++))` returns exit 1 the first time COUNTER is 0, which
#     kills the script under `set -e`. We use COUNTER=$((COUNTER+1))
#     instead.
#   - `pipe | while read; done` runs the loop body in a subshell, so any
#     counter updates inside the loop don't survive. We use process
#     substitution (`done < <(...)`) instead.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

NEW_VER="${1:-}"
if [[ -z "$NEW_VER" ]]; then
    echo "Usage: $0 <NEW_VER>"
    echo "Example: $0 0.9.6.0"
    exit 1
fi

INVENTORY_FILE="VERSION_INVENTORY.csv"
VERSION_FILE="VERSION"

if [[ ! -f "$INVENTORY_FILE" ]]; then
    echo "Error: $INVENTORY_FILE not found." >&2
    echo "Please run scripts/version-inventory.sh first." >&2
    exit 1
fi

# Whitelist of known-legacy MORIE versions safe to overwrite.
# Extend this when you find a stale stamp the regular CURRENT pass missed.
KNOWN_LEGACY_VERSIONS=("0.3.0" "0.2.0")

# 1. Update the canonical version file.
echo "$NEW_VER" > "$VERSION_FILE"
echo "==> Updated $VERSION_FILE to $NEW_VER"
echo "==> Applying patches..."

PATCH_COUNT=0
SKIP_COUNT=0

# 2. Walk the CSV and patch targeted lines.
#    Process-substitution (< <(...)) keeps PATCH_COUNT/SKIP_COUNT live.
while IFS=, read -r file line status old_ver context; do
    old_ver=$(echo "$old_ver" | tr -d '\r')
    status=$(echo "$status" | tr -d '\r')

    if [[ ! -f "$file" ]]; then
        echo "  [!] Skipping $file (file not found)"
        continue
    fi

    SHOULD_PATCH=false
    if [[ "$status" == "CURRENT" ]]; then
        SHOULD_PATCH=true
    elif [[ "$status" == "STALE_OR_DEP" ]]; then
        for legacy_ver in "${KNOWN_LEGACY_VERSIONS[@]}"; do
            if [[ "$old_ver" == "$legacy_ver" ]]; then
                SHOULD_PATCH=true
                break
            fi
        done
    fi

    if [[ "$SHOULD_PATCH" == true ]]; then
        # Targeted in-place edit on a specific line number.
        # \Q...\E protects regex metacharacters in $old_ver.
        perl -pi -e "if (\$. == $line) { s/\b\Q$old_ver\E\b/$NEW_VER/g }" "$file"
        echo "  [x] Patched $file:$line ($old_ver -> $NEW_VER)"
        PATCH_COUNT=$((PATCH_COUNT + 1))
    else
        SKIP_COUNT=$((SKIP_COUNT + 1))
    fi
done < <(tail -n +2 "$INVENTORY_FILE")

echo "==> Done. Applied $PATCH_COUNT patches. Skipped $SKIP_COUNT rows (dependencies / out-of-scope)."

# 3. Auto-regenerate so the CSV stays in sync.
echo "==> Regenerating inventory for verification..."
./scripts/version-inventory.sh
