# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Baseline-corrected correlation."""

from __future__ import annotations

from . import _array_core as np
from ._containers import DescriptiveResult


def baseline_corrected_correlation(x: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """Compute Pearson correlation after baseline (mean) subtraction."""
    from morie._waveform import baseline_corrected_correlation as _backend

    corr = _backend(x, y)
    return DescriptiveResult(
        name="baseline_corrected_corr",
        value=corr,
        extra={"correlation": corr},
    )


bscor = baseline_corrected_correlation


def cheatsheet() -> str:
    return "baseline_corrected_correlation({}) -> Baseline-corrected correlation."
