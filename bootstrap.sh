#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# bootstrap.sh -- offer to install the rest of the morie family.
#
# The morie family spans two open ecosystems plus one proprietary CLI:
#   * morie            -- Python package (PyPI)            [OPEN, AGPL]
#   * rmorie           -- R package (r-universe)           [OPEN, AGPL]
#       + rmoriedata, rmoriebricklayer install automatically as rmorie deps
#   * rmorie-cli       -- C++ binary, Receipt-of-Custody   [PROPRIETARY]
#
# This script detects what is already present and offers to install the
# MISSING OPEN packages. The proprietary CLI is never auto-installed --
# we only point to where to obtain it.
#
# Run directly:
#   curl -fsSL https://rootcoder007.github.io/morie/bootstrap.sh | sh
# Flags:
#   -y, --yes     install without prompting (assume yes)
#       --check   detect + report only; install nothing
#   -h, --help    this help
# Env:
#   MORIE_NO_BOOTSTRAP=1   exit immediately, do nothing
#   MORIE_BOOTSTRAP_YES=1  same as --yes
#
# Exit: 0 ok / nothing to do, 1 an install failed, 2 bad usage.
set -eu

ASSUME_YES=0
CHECK_ONLY=0
[ "${MORIE_BOOTSTRAP_YES:-0}" = "1" ] && ASSUME_YES=1

if [ "${MORIE_NO_BOOTSTRAP:-0}" = "1" ]; then
    exit 0
fi

for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        --check)  CHECK_ONLY=1 ;;
        -h|--help)
            sed -n '3,30p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "bootstrap: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

CLI_URL="https://github.com/rootcoder007/rmorie-cli"
RUNIV="https://rootcoder007.r-universe.dev"
CRAN="https://cloud.r-project.org"

# ---- detection ------------------------------------------------------------

find_python() {
    for p in python3 python; do
        if command -v "$p" >/dev/null 2>&1; then echo "$p"; return 0; fi
    done
    return 1
}

PY=$(find_python || true)

have_py_morie() {
    [ -n "$PY" ] && "$PY" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("morie") else 1)' >/dev/null 2>&1
}
have_rscript() { command -v Rscript >/dev/null 2>&1; }
have_r_morie() {
    have_rscript && Rscript -e 'quit(status = as.integer(!requireNamespace("rmorie", quietly = TRUE)))' >/dev/null 2>&1
}
have_cli() { command -v rmorie >/dev/null 2>&1; }

# The morie family is built on a shared C/C++ numeric core (libmorie ->
# morie._core in Python; rmoriebricklayer's compiled kernels in R). Without
# a C/C++ toolchain the packages either fail to build from source or fall
# back to slow pure-language kernels -- so we detect the toolchain and
# verify the compiled backend actually loaded, and warn if not.
have_toolchain() {
    { command -v cc || command -v gcc || command -v clang; } >/dev/null 2>&1 &&
    { command -v c++ || command -v g++ || command -v clang++; } >/dev/null 2>&1
}
# Compiled backend actually active?
py_backend_ok() {
    [ -n "$PY" ] && "$PY" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("morie._core") else 1)' >/dev/null 2>&1
}
r_backend_ok() {
    have_rscript && Rscript -e 'quit(status = as.integer(!isTRUE(rmorie::morie_fast_available())))' >/dev/null 2>&1
}

# ---- report ---------------------------------------------------------------

mark() { if [ "$1" = 0 ]; then printf '  [x] %s\n' "$2"; else printf '  [ ] %s\n' "$2"; fi; }

py_ok=1; r_ok=1; cli_ok=1
have_py_morie && py_ok=0
have_r_morie  && r_ok=0
have_cli      && cli_ok=0

echo "morie family status:"
mark "$py_ok"  "morie            (Python / pip)"
mark "$r_ok"   "rmorie + data + bricklayer  (R / r-universe)"
mark "$cli_ok" "rmorie-cli       (proprietary -- not auto-installed)"

# The whole family is built on a shared C/C++ core. Surface it plainly:
# without the toolchain the packages do not work properly.
tc_ok=1; have_toolchain && tc_ok=0
mark "$tc_ok" "C/C++ toolchain  (cc + c++ -- REQUIRED for the compiled core)"
if [ "$py_ok" = 0 ] && ! py_backend_ok; then
    echo "  !! morie is installed but its C++ backend (morie._core) is NOT active -- degraded."
fi
if [ "$r_ok" = 0 ] && ! r_backend_ok; then
    echo "  !! rmorie is installed but its C/C++ kernels are NOT active (fast unavailable) -- slow."
fi
if [ "$tc_ok" = 1 ]; then
    cat <<'EOF'
  !! No C/C++ toolchain detected. morie and rmorie are built on a shared
     C/C++ core; without a compiler they fall back to slow pure-language
     kernels (or fail to build from source). Install one FIRST:
       Debian/Ubuntu : sudo apt-get install build-essential
       macOS         : xcode-select --install
       Fedora/RHEL   : sudo dnf install gcc gcc-c++ make
EOF
fi
echo

if [ "$CHECK_ONLY" = 1 ]; then exit 0; fi

# What is missing + installable (open packages only)?
need_py=0; need_r=0
[ "$py_ok" = 1 ] && [ -n "$PY" ] && need_py=1
[ "$r_ok" = 1 ] && have_rscript && need_r=1

# Surface uninstallable-here situations without failing.
[ "$py_ok" = 1 ] && [ -z "$PY" ] && echo "note: Python not found -- install Python first, then 'pip install morie'."
[ "$r_ok" = 1 ] && ! have_rscript && echo "note: R not found -- install R first, then re-run this bootstrap."
[ "$cli_ok" = 1 ] && echo "note: rmorie-cli is proprietary (Receipt-of-Custody); obtain it at $CLI_URL"

if [ "$need_py" = 0 ] && [ "$need_r" = 0 ]; then
    echo "Nothing to install. The open morie family is already present."
    exit 0
fi

# ---- prompt ---------------------------------------------------------------

if [ "$ASSUME_YES" != 1 ]; then
    if [ ! -t 0 ]; then
        echo "Non-interactive shell; not installing. Re-run with --yes, or:"
        [ "$need_py" = 1 ] && echo "  $PY -m pip install morie"
        [ "$need_r" = 1 ]  && echo "  Rscript -e \"install.packages('rmorie', repos=c('$RUNIV','$CRAN'))\""
        exit 0
    fi
    printf 'Install the missing open package(s) now? [Y/n] '
    read -r reply || reply=""
    case "$reply" in
        ''|y|Y|yes|YES) : ;;
        *) echo "Skipped. You can re-run anytime."; exit 0 ;;
    esac
fi

# ---- install (in parallel for speed) --------------------------------------

rc=0
pid_py=""; pid_r=""
if [ "$need_py" = 1 ]; then
    ( echo "-> installing Python morie ..."; "$PY" -m pip install --upgrade morie ) &
    pid_py=$!
fi
if [ "$need_r" = 1 ]; then
    ( echo "-> installing R rmorie (+ rmoriedata, rmoriebricklayer) ..."; \
      Rscript -e "install.packages('rmorie', repos=c('$RUNIV','$CRAN'))" ) &
    pid_r=$!
fi

[ -n "$pid_py" ] && { wait "$pid_py" || rc=1; }
[ -n "$pid_r" ]  && { wait "$pid_r"  || rc=1; }

echo
if [ "$rc" = 0 ]; then
    echo "Done. morie family installed."
    # Installed != working: verify the shared C/C++ backend actually loaded.
    if [ "$need_py" = 1 ] && ! py_backend_ok; then
        echo "WARNING: morie installed, but its C++ backend (morie._core) did NOT load --"
        echo "         it will be degraded. Install a C/C++ toolchain, then reinstall:"
        echo "           $PY -m pip install --force-reinstall --no-binary :all: morie"
    fi
    if [ "$need_r" = 1 ] && ! r_backend_ok; then
        echo "WARNING: rmorie installed, but its C/C++ kernels are INACTIVE"
        echo "         (morie_fast_available() is FALSE) -- it will be slow. Install a"
        echo "         C/C++ toolchain, then reinstall rmorie."
    fi
else
    echo "One or more installs failed (see output above)." >&2
fi
exit "$rc"
