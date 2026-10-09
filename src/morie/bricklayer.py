# SPDX-License-Identifier: AGPL-3.0-or-later
"""``morie bricklayer`` / ``morie r-install`` -- offer to install the rest of
the morie family.

Python is already present (you ran ``morie``), so this focuses on the
*other* ecosystems:

* R: ``rmorie``, ``rmoriedata`` and ``rmoriebricklayer``, all from
  r-universe (CRAN carries older companions).
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

    Default: rmorie from r-universe (prebuilt binaries), with its companions
    rmoriebricklayer and rmoriedata named explicitly. ``github``: this
    repository's own R arm, built from source with remotes (needs a C/C++
    toolchain).

    Every call carries ``repos``: under ``Rscript`` there is no mirror
    chooser, so a bare ``install.packages()`` stops with "trying to use CRAN
    without setting a mirror". r-universe comes first because CRAN carries
    older companions, and the companions are named (and remotes upgrades
    ``"always"``) because an older copy already installed satisfies a
    dependency check and would otherwise be kept.
    """
    from . import __version__ as v

    repos = f"c('{RUNIV}','{CRAN}')"
    companions = f"install.packages(c('rmoriebricklayer','rmoriedata'), repos={repos}); "
    remotes = f"if (!requireNamespace(\"remotes\", quietly = TRUE)) install.packages('remotes', repos='{CRAN}'); "

    if github:
        # pinned to this release's tag, as the r-universe route is: the default branch
        # holds whatever release main is on (1.3.9 while 1.4.0 was on lean), and the
        # R bridge refuses a mismatched arm. A dev build takes main.
        import re

        ref = f"@v{v}" if re.match(r"^\d+\.\d+\.\d+$", v) else ""
        return (
            companions + remotes + f"remotes::install_github('{GITHUB_REPO}{ref}', subdir = '{GITHUB_SUBDIR}', "
            f"repos = {repos}, upgrade = 'always')"
        )

    # the R arm must be the same release as morie: r-universe first, its release tag when r-universe
    # serves another version (it lags a release by a build cycle)
    return (
        f"install.packages(c('rmoriebricklayer','rmoriedata','rmorie'), repos={repos}); "
        "have <- function() tryCatch(as.character(utils::packageVersion('rmorie')), error = function(e) ''); "
        f"if (have() != '{v}') {{ "
        + remotes
        + f"remotes::install_github('rootcoder007/rmorie@v{v}', repos = {repos}, upgrade = 'always') }}; "
        f"if (have() != '{v}') stop('rmorie {v} is not published yet')"
    )


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
    return _r_eval_ok("quit(status = as.integer(!isTRUE(rmorie::morie_fast_available())))")


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
        help="Offer to install the rest of the morie family (R packages) and verify the shared C/C++ backend.",
    )
    _add_install_args(p)
    r = subparsers.add_parser(
        "r-install",
        help="Install the R side: rmorie from r-universe, or this repository's own R arm from GitHub with --github.",
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
    """Run ``morie bricklayer``: report which members of the morie family are installed.

    Prints one line each for morie (Python), rmorie with its data and
    bricklayer packages (R), the ``rmorie`` launcher and a C/C++ toolchain,
    and warns when a compiled core is inactive. With ``args.check`` it
    only reports. Otherwise, when the R side is missing (or
    ``args.github`` asks for this repository's R arm), it installs it --
    rmorie pinned to this morie's version -- after a confirmation that
    ``args.yes`` skips; off a terminal without ``--yes`` it prints the
    install command instead.

    Args:
        args: the parsed ``argparse`` namespace of the ``bricklayer`` verb.

    Returns:
        The process exit code (0 when everything checked is present).
    """
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
        print(
            "note: the rmorie launcher is not on PATH; put it there with "
            "Rscript -e 'rmorie::install_cli()' (then: rmorie login, rmorie models, rmorie ask ...)"
        )

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

    from ._progress import run_step

    rc = run_step(
        [_rscript(), "-e", expr],
        "installing the R side from GitHub (compiles; minutes)"
        if github
        else "installing rmorie, rmoriedata, rmoriebricklayer",
    )

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
