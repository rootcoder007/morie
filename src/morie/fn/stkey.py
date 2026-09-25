"""Siegel-Tukey test for scale equality."""

from . import _array_core as np
from .gb941 import sgltukey

__all__ = ["stkey"]


def stkey(x, y, axis=0, cdf=None):
    r"""
    Siegel-Tukey test for equality of scale parameters.

    The pooled sample is scored from the ends inwards -- 1 to the
    smallest, 2 and 3 to the two largest, 4 and 5 to the next two
    smallest, and so on -- and the X scores are summed; with N odd the
    middle observation is dropped first (Siegel and Tukey 1960; Gibbons
    and Chakraborti 2011, Sec. 9.4). The z statistic uses the exact
    linear-rank moments. This is :func:`morie.fn.gb941.sgltukey`, whose
    p-value matches DescTools::SiegelTukeyTest(exact = FALSE,
    correct = FALSE).

    The previous version dealt the scores by the parity of each value's
    sorted position, so the 2nd, 4th, ... smallest values received the
    largest scores, and it used n_x n_y (n+1)^2 / (12 (n-1)) as the
    variance of a sum of ranks, which is n_x n_y (n+1) / 12.

    `cdf` is unused and kept for signature stability.
    """
    del cdf
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 2:
        x = np.take(x, 0, axis=axis)
    if y.ndim == 2:
        y = np.take(y, 0, axis=axis)
    if len(x) < 1 or len(y) < 1:
        raise ValueError("Both samples must have ≥1 observation")
    r = sgltukey(x.tolist(), y.tolist())
    p_value = r["p_value"]
    return {
        "statistic": float(r["statistic"]),
        "z_stat": float(r["z"]),
        "p_value": float(p_value),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }


def cheatsheet() -> str:
    return "stkey: stkey(x, y, axis, cdf) -> Siegel-Tukey test for equality of scale parameters."
