"""Peak-to-peak amplitude."""

from __future__ import annotations

from . import _array_core as np

from ._containers import DescriptiveResult

_QUOTE = "Stay on target."


def peak_to_peak(x, **kwargs) -> DescriptiveResult:
    r"""Compute the peak-to-peak amplitude of signal *x*.

    .. math::

        \\text{pp} = \\max(x) - \\min(x)

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    pp = float(np.max(x) - np.min(x))
    return DescriptiveResult(
        name="peak_to_peak",
        value=pp,
        extra={"peak_to_peak": pp, "max": float(np.max(x)), "min": float(np.min(x)), "n": len(x)},
    )


spp = peak_to_peak


def cheatsheet() -> str:
    return "peak_to_peak({}) -> Peak-to-peak amplitude."


# compact alias per ledger/NAMING.md
peaktopeak = peak_to_peak
