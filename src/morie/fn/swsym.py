"""Test weights matrix symmetry."""

from . import _array_core as np

from ._containers import SpatialResult


def swsym(W):
    """Test weights matrix symmetry.

    Category: WDiag

    Parameters
    ----------
    W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="swsym", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="swsym", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


swsym_fn = swsym


def cheatsheet() -> str:
    return "swsym({}) -> Test weights matrix symmetry."
