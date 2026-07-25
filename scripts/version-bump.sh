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

# Files whose version strings are HISTORY, never a current stamp.
#
# Release notes record what a past release did ("# morie 1.1.4 - 2026-07-15",
# "CITATION.cff version resynced (had lagged at 1.1.4)"). Rewriting those is
# always wrong: it renames a shipped release and turns true sentences false.
# On 2026-07-25 this script did exactly that, retitling the 1.1.4 section as
# 1.1.7 while keeping 1.1.4's date. Add entries here rather than patching.
NEVER_PATCH_RE='(^|/)(NEWS\.md|CHANGELOG\.md|WHATS_NEW\.md|.*_tracker\.md)$'

# 1. Establish the TRUE current version before touching anything.
#
# version-inventory.sh classifies a row CURRENT when it matches the VERSION
# file, so a stale VERSION silently inverts every decision this script makes:
# the real version gets marked STALE_OR_DEP and is never bumped, while
# historical mentions of the stale value get marked CURRENT and are rewritten.
# That is precisely how 1.1.6 shipped with pyproject.toml unbumped and NEWS.md
# corrupted. Trust the release manifests, not the VERSION file.
read_ver() { [[ -f "$1" ]] && sed -nE "s/$2/\1/p" "$1" | head -1 || true; }

PYPROJECT_VER=$(read_ver pyproject.toml '^version[[:space:]]*=[[:space:]]*"([^"]+)".*')
CITATION_VER=$(read_ver CITATION.cff '^version:[[:space:]]*"?([^"[:space:]]+)"?.*')
FILE_VER=$(tr -d '[:space:]' < "$VERSION_FILE" 2>/dev/null || true)

if [[ -z "$PYPROJECT_VER" ]]; then
    echo "Error: could not read version from pyproject.toml." >&2
    exit 1
fi

if [[ -n "$CITATION_VER" && "$CITATION_VER" != "$PYPROJECT_VER" ]]; then
    echo "Error: release manifests disagree, refusing to guess." >&2
    echo "  pyproject.toml: $PYPROJECT_VER" >&2
    echo "  CITATION.cff:   $CITATION_VER" >&2
    echo "Reconcile them by hand, then re-run." >&2
    exit 1
fi

OLD_VER="$PYPROJECT_VER"
echo "==> Current version (from release manifests): $OLD_VER"

if [[ "$FILE_VER" != "$OLD_VER" ]]; then
    echo "==> WARNING: $VERSION_FILE said '$FILE_VER' but the real version is '$OLD_VER'."
    echo "    Self-healing $VERSION_FILE and regenerating the inventory so that"
    echo "    CURRENT/STALE_OR_DEP are computed against the real version."
    echo "$OLD_VER" > "$VERSION_FILE"
    ./scripts/version-inventory.sh >/dev/null
fi

if [[ "$NEW_VER" == "$OLD_VER" ]]; then
    echo "Error: requested version $NEW_VER equals the current version." >&2
    exit 1
fi

# 2. Patch the release manifests explicitly.
#
# These must be bumped whether or not the inventory happened to list them; a
# missing or misclassified CSV row is not permitted to skip a release file.
echo "==> Patching release manifests..."
perl -pi -e "s/^version(\s*=\s*)\"\Q$OLD_VER\E\"/version\${1}\"$NEW_VER\"/" pyproject.toml
echo "  [x] pyproject.toml ($OLD_VER -> $NEW_VER)"
if [[ -f CITATION.cff ]]; then
    perl -pi -e "s/^version:(\s*)\"?\Q$OLD_VER\E\"?/version:\${1}\"$NEW_VER\"/" CITATION.cff
    echo "  [x] CITATION.cff ($OLD_VER -> $NEW_VER)"
fi
echo "$NEW_VER" > "$VERSION_FILE"
echo "  [x] $VERSION_FILE ($OLD_VER -> $NEW_VER)"

# Fail loudly rather than shipping a half-bumped release.
for f in pyproject.toml CITATION.cff; do
    [[ -f "$f" ]] || continue
    if ! grep -qF "$NEW_VER" "$f"; then
        echo "Error: $f does not contain $NEW_VER after patching." >&2
        exit 1
    fi
done

echo "==> Applying inventory-driven patches..."

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

    # Release notes and trackers record history; never rewrite them.
    if [[ "$file" =~ $NEVER_PATCH_RE ]]; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # Already handled explicitly above.
    case "$file" in
        ./pyproject.toml|pyproject.toml|./CITATION.cff|CITATION.cff|./VERSION|VERSION)
            SKIP_COUNT=$((SKIP_COUNT + 1)); continue ;;
    esac

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
