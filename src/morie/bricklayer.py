# SPDX-License-Identifier: AGPL-3.0-or-later
"""``morie bricklayer`` / ``morie r-install`` -- offer to install the rest of
the morie family.

Python is already present (you ran ``morie``), so this focuses on the
*other* ecosystems:

* R: ``rmorie`` from r-universe, which pulls ``rmoriedata`` +
  ``rmoriebricklayer`` as dependencies.
* the ``rmorie`` command-line launcher ships inside rmorie;
  ``rmorie::install_cli()`` links it onto PATH.

The whole family is built on a shared C/C++ numeric core (``libmorie`` ->
``morie._core`` in Python; ``rmoriebricklayer``'s compiled kernels in R).
This command verifies that core actually loaded and warns -- loudly -- that
the project does not work properly without a C/C++ toolchain.

Pure stdlib, cross-platform (works for Windows pip users with no shell).
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys

RUNIV = "https://rootcoder007.r-universe.dev"
CRAN = "https://cloud.r-project.org"
GITHUB_REPO = "rootcoder007/morie"
GITHUB_SUBDIR = "r-package/morie"


def _r_install_expr(github: bool = False) -> str:
    """The R expression that installs the R side.

    Default: rmorie from r-universe (prebuilt binaries, pulls rmoriedata and
    rmoriebricklayer). ``github``: this repository's own R arm, built from
    source with remotes (needs a C/C++ toolchain and rmoriebricklayer).
    """
    if github:
        return (
            'if (!requireNamespace("remotes", quietly = TRUE)) '
            f"install.packages('remotes', repos='{CRAN}'); "
            f"remotes::install_github('{GITHUB_REPO}', subdir = '{GITHUB_SUBDIR}')"
        )
    return f"install.packages('rmorie', repos=c('{RUNIV}','{CRAN}'))"


def _have_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _rscript() -> str | None:
    return shutil.which("Rscript")


def _r_eval_ok(expr: str) -> bool:
    rs = _rscript()
    if not rs:
        return False
    try:
        return subprocess.run([rs, "-e", expr], capture_output=True).returncode == 0
    except OSError:
        return False


def _have_r_morie() -> bool:
    return _r_eval_ok('quit(status = as.integer(!requireNamespace("rmorie", quietly = TRUE)))')


def _r_backend_ok() -> bool:
    return _r_eval_ok('quit(status = as.integer(!isTRUE(rmorie::morie_fast_available())))')


def _py_backend_ok() -> bool:
    """Is morie's compiled C++ core (morie._core) importable?"""
    return _have_spec("morie._core")


def _have_toolchain() -> bool:
    cc = any(_which(c) for c in ("cc", "gcc", "clang"))
    cxx = any(_which(c) for c in ("c++", "g++", "clang++"))
    return cc and cxx


def register_subparser(subparsers) -> None:
    """Called from morie.runner.build_parser() to register this command."""
    p = subparsers.add_parser(
        "bricklayer",
        help="Offer to install the rest of the morie family (R packages) and "
        "verify the shared C/C++ backend.",
    )
    _add_install_args(p)
    r = subparsers.add_parser(
        "r-install",
        help="Install the R side: rmorie from r-universe, or this repository's "
        "own R arm from GitHub with --github.",
    )
    _add_install_args(r)


def _add_install_args(p) -> None:
    p.add_argument("-y", "--yes", action="store_true", help="install without prompting")
    p.add_argument("--check", action="store_true", help="report status only; install nothing")
    p.add_argument(
        "--github",
        action="store_true",
        help="build r-package/morie from GitHub with remotes instead of installing "
        "rmorie from r-universe (needs a C/C++ toolchain and rmoriebricklayer)",
    )


def _mark(ok: bool, label: str) -> None:
    print(f"  [{'x' if ok else ' '}] {label}")


def run(args) -> int:
    py_ok = True  # we are running inside morie
    r_ok = _have_r_morie()
    cli_ok = _which("rmorie")
    tc_ok = _have_toolchain()

    print("morie family status:")
    _mark(py_ok, "morie            (Python / this interpreter)")
    _mark(r_ok, "rmorie + data + bricklayer  (R / r-universe)")
    _mark(cli_ok, "rmorie launcher  (rmorie on PATH; Rscript -e 'rmorie::install_cli()')")
    _mark(tc_ok, "C/C++ toolchain  (cc + c++ -- REQUIRED for the compiled core)")

    if not _py_backend_ok():
        print("  !! morie's C++ backend (morie._core) is NOT active -- this install is degraded.")
    if r_ok and not _r_backend_ok():
        print("  !! rmorie is installed but its C/C++ kernels are inactive (slow pure-R fallback).")
    if not tc_ok:
        print(
            "  !! No C/C++ toolchain detected. morie and rmorie are built on a shared\n"
            "     C/C++ core; without a compiler they fall back to slow pure-language\n"
            "     kernels (or fail to build from source). Install one FIRST:\n"
            "       Debian/Ubuntu : sudo apt-get install build-essential\n"
            "       macOS         : xcode-select --install\n"
            "       Fedora/RHEL   : sudo dnf install gcc gcc-c++ make"
        )
    print()

    if getattr(args, "check", False):
        return 0

    if r_ok and not cli_ok:
        print("note: the rmorie launcher is not on PATH; put it there with "
              "Rscript -e 'rmorie::install_cli()' (then: rmorie login, rmorie models, rmorie ask ...)")

    if r_ok and not getattr(args, "github", False):
        print("Nothing to install: the R side is already present.")
        return 0

    github = bool(getattr(args, "github", False))
    expr = _r_install_expr(github)
    if not _rscript():
        print("R is not installed. Install R first (https://cloud.r-project.org), then:")
        print(f'  Rscript -e "{expr}"')
        return 0

    if not getattr(args, "yes", False):
        if not sys.stdin.isatty():
            print("Non-interactive; not installing. Re-run with --yes, or:")
            print(f'  Rscript -e "{expr}"')
            return 0
        what = (
            "this repository's R arm (r-package/morie) from GitHub"
            if github
            else "the R package rmorie (+ rmoriedata, rmoriebricklayer)"
        )
        reply = input(f"Install {what} now? [Y/n] ").strip()
        if reply not in ("", "y", "Y", "yes", "YES"):
            print("Skipped. Re-run `morie bricklayer` anytime.")
            return 0

    print("-> installing the R side ..." if github else "-> installing R rmorie (+ rmoriedata, rmoriebricklayer) ...")
    rc = subprocess.run([_rscript(), "-e", expr]).returncode

    if rc != 0:
        print("R install failed (see output above).", file=sys.stderr)
        return rc

    print("\nDone.")
    if not _r_backend_ok():
        print(
            "WARNING: rmorie installed, but its C/C++ kernels are INACTIVE "
            "(morie_fast_available() is FALSE) -- it will be slow. Install a "
            "C/C++ toolchain and reinstall rmorie."
        )
    return 0
