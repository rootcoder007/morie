#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# version-inventory.sh — regenerate VERSION_INVENTORY.csv deterministically.
#
# Single source of truth: the root-level VERSION file.
# Anything else that mentions a version number anywhere in the repo is
# *derived* — this script scans the tree, compares each hit against the
# canonical VERSION, and writes a categorised CSV.
#
# DRY contract:
#   - VERSION is the only file you hand-edit when bumping the package.
#   - This script regenerates VERSION_INVENTORY.csv from current sources.
#   - scripts/version-bump.sh propagates a new VERSION into all CURRENT
#     locations (and a small whitelist of known-legacy MORIE versions),
#     then re-runs this script for verification.
#   - A CI workflow runs this script and fails if the committed CSV drifts
#     from the regenerated output (see .github/workflows/version-drift.yml).
#
# Design notes:
#   - rg is used with --vimgrep for predictable `file:line:col:context`
#     output. ripgrep's gitignore handling automatically excludes .venv,
#     node_modules, target/, etc.
#   - We blackhole src/morie/fn/describe_*.md from the scan because those
#     files contain textbook section numbers (5.4.4, 2.11.2, …) that look
#     like versions and produced ~90 false positives in the previous CSV.

set -euo pipefail

# Force execution from the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

INVENTORY_FILE="VERSION_INVENTORY.csv"
VERSION_FILE="VERSION"

# 1. Ensure canonical source exists.
if [[ ! -f "$VERSION_FILE" ]]; then
    echo "Error: Canonical $VERSION_FILE not found." >&2
    echo "Please create it containing only the current version (e.g., 0.9.5.5)." >&2
    exit 1
fi

CURRENT_VERSION=$(tr -d '[:space:]' < "$VERSION_FILE")

echo "==> Scanning for versions (canonical target: $CURRENT_VERSION)..."

# 2. Match 3 or 4 segment version numbers (e.g., 0.2.0, 0.9.5.5).
REGEX='[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?'

# 3. Initialise CSV with header.
echo "File,Line,Status,Version,Context" > "$INVENTORY_FILE"

# 4. ripgrep pipeline.
#    --vimgrep gives standard 'file:line:column:context'
#    -g flags handle exclusions and prevent self-scanning.
rg --vimgrep --color=never -e "$REGEX" \
  -g '!src/morie/fn/describe_*.md' \
  -g '!VERSION_INVENTORY.csv' \
  -g '!scripts/' \
  -g '!.git/' \
  . | awk -v curr="$CURRENT_VERSION" -F':' '{
    file = $1
    line = $2

    # Reconstruct the code context (preserve colons inside the line).
    context = $4
    for (i = 5; i <= NF; i++) context = context ":" $i

    # Strip double-quotes and commas to avoid CSV escaping nightmares.
    gsub(/"/, "", context)
    gsub(/,/, ";", context)

    # Trim leading/trailing whitespace.
    gsub(/^[ \t]+|[ \t]+$/, "", context)

    # Extract the actual version string from the context line.
    if (match(context, /[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?/)) {
        matched_ver = substr(context, RSTART, RLENGTH)

        # DOI fragments (10.1080/01621459.1992.10475217) match the semver
        # regex but the leading segment is 7-8 digits, not 1-3.  Skip
        # anything with a 4+ digit major.
        split(matched_ver, parts, ".")
        if (length(parts[1]) >= 4) next

        if (matched_ver == curr) {
            status = "CURRENT"
        } else {
            status = "STALE_OR_DEP"
        }

        printf "%s,%s,%s,%s,\"%s\"\n", file, line, status, matched_ver, context
    }
}' >> "$INVENTORY_FILE"

# Console summary.
TOTAL_ROWS=$(tail -n +2 "$INVENTORY_FILE" | wc -l | tr -d ' ')
STALE_ROWS=$(grep -c ",STALE_OR_DEP," "$INVENTORY_FILE" || true)

echo "==> Done. Generated $INVENTORY_FILE"
echo "    Found $TOTAL_ROWS version mentions ($STALE_ROWS stale or dependency)."
