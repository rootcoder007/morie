# morie.fn -- function file (rootcoder007/morie)
"""Ruler fractal dimension."""

from __future__ import annotations

from . import _array_core as np
from ._containers import DescriptiveResult


def ruler_fd(x: np.ndarray, n_rulers: int = 10) -> DescriptiveResult:
    """Ruler (divider) fractal dimension of a signal curve over ``n_rulers`` ruler lengths."""
    from morie._waveform import ruler_fd as _backend

    fd = _backend(x, n_rulers=n_rulers)
    return DescriptiveResult(
        name="ruler_fd",
        value=fd,
        extra={"fd": fd, "n_rulers": n_rulers},
    )


rulfd = ruler_fd


def cheatsheet() -> str:
    return "ruler_fd({}) -> Ruler fractal dimension."


# compact alias per ledger/NAMING.md
rulerfd = ruler_fd
