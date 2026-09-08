"""Bartlett (triangular) window."""

from __future__ import annotations

from . import _array_core as np

from ._containers import DescriptiveResult

_QUOTE = "Never underestimate a droid."


def bartlett_window(N: int, **kwargs) -> DescriptiveResult:
    """Generate a Bartlett (triangular) window of length *N*.

    Parameters
    ----------
    N : int
        Window length.

    Returns
    -------
    DescriptiveResult
    """
    w = np.bartlett(N)
    return DescriptiveResult(
        name="bartlett_window",
        value=float(N),
        extra={"window": w, "N": N},
    )


wnbrt = bartlett_window


def cheatsheet() -> str:
    return "bartlett_window({}) -> Bartlett (triangular) window."


# compact alias per ledger/NAMING.md
bartlettwindow = bartlett_window
