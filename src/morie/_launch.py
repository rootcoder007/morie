# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every program morie starts goes through this module (part of the interactive layer).

R (Rscript), Docker, the editor, the C compiler for the optional kernels, gcloud, pip,
ssh, quarto/jupyter and the file viewer are all started from here. The module stays
out of the published wheel and sdist, so the installed package itself never launches
another program; ``morie interactive install`` adds it, and
:func:`morie._interactive.launcher` hands it to the code that needs it.

It is the standard :mod:`subprocess` interface, re-exported unchanged, so a call site
reads ``sp = launcher("R"); sp.run([...])`` exactly as it did with ``subprocess``.
"""

from __future__ import annotations

import subprocess
from subprocess import (  # noqa: F401 - re-exported for call sites
    DEVNULL,
    PIPE,
    STDOUT,
    CalledProcessError,
    CompletedProcess,
    Popen,
    SubprocessError,
    TimeoutExpired,
    call,
    check_call,
    check_output,
    run,
)


def r_cmd(expr: str, *, rscript: str = "Rscript", options: tuple[str, ...] = ()) -> list[str]:
    """The command line that runs one R expression: ``[rscript, *options, "-e", expr]``."""
    return [rscript, *options, "-e", expr]


def r_expr(expr: str, *, rscript: str = "Rscript", options: tuple[str, ...] = (), **kwargs):
    """Run one R expression: ``Rscript [options] -e expr``, with ``subprocess.run``'s keyword arguments.

    Every call site passes a fixed expression (a version probe, an install call); none builds one
    from user input.
    """
    return run(r_cmd(expr, rscript=rscript, options=options), **kwargs)


__all__ = [
    "DEVNULL",
    "PIPE",
    "STDOUT",
    "CalledProcessError",
    "CompletedProcess",
    "Popen",
    "SubprocessError",
    "TimeoutExpired",
    "call",
    "check_call",
    "check_output",
    "r_cmd",
    "r_expr",
    "run",
    "subprocess",
]
