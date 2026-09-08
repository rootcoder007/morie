"""Shape factor of a waveform."""

from __future__ import annotations

from . import _array_core as np

from ._containers import DescriptiveResult


def shape_factor_fn(x: np.ndarray) -> DescriptiveResult:
    """Compute the shape factor (RMS / mean of absolute value).

    'The whole is greater than the sum of its parts.' -- Aristotle
    """
    from morie._waveform import shape_factor as _backend

    result = _backend(x)
    return DescriptiveResult(name="shape_factor", value=float(result))


alias = shape_factor_fn


def cheatsheet() -> str:
    return "shape_factor_fn({}) -> Shape factor of a waveform."


# compact alias per ledger/NAMING.md
shapefactorfn = shape_factor_fn
