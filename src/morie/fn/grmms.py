# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Min-max scaling to [0, 1]."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_minmax_scaler"]

_METHOD = "Min-max feature scaling"


def geron_minmax_scaler(X, feature_range=(0.0, 1.0)):
    r"""Rescale each column into ``[0, 1]``.

    .. math::
        x_{\text{scaled}} = \frac{x - \min(x)}{\max(x) - \min(x)}

    Per column, never over the whole matrix: features are on different
    units, and the point of the transform is to remove exactly that
    difference.  A constant column has zero range and cannot be scaled
    -- that raises here rather than returning ``nan`` or silently
    mapping everything to zero, because a constant feature is a data
    problem worth seeing.

    The fitted ``data_min`` / ``data_range`` are returned so the same
    transform can be replayed on a test set; refitting on test data is
    the classic leak.

    Parameters
    ----------
    X : array-like, shape (m,) or (m, n)
    feature_range : (float, float), optional
        Target interval, default ``(0, 1)``.

    Returns
    -------
    RichResult
        Payload keys ``scaled``, ``data_min``, ``data_max``,
        ``data_range``, ``scale``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 2, Feature Scaling section (min-max).

    Examples
    --------
    Column of 1, 3, 5 maps to 0, 0.5, 1:

    >>> r = geron_minmax_scaler([[1.0], [3.0], [5.0]])
    >>> r["scaled"]
    [[0.0], [0.5], [1.0]]
    >>> r["data_range"]
    [4.0]

    Each column is scaled on its own range, so wildly different units
    end up comparable:

    >>> r2 = geron_minmax_scaler([[0.0, 100.0], [1.0, 300.0]])
    >>> r2["scaled"]
    [[0.0, 0.0], [1.0, 1.0]]
    """
    A = np.asarray(X, dtype=float)
    vector = A.ndim == 1
    if vector:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got ndim {A.ndim}.")
    if A.shape[0] == 0:
        raise ValueError("X has no rows.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X must be finite; impute missing values first (see grimp).")
    lo, hi = (float(v) for v in feature_range)
    if not hi > lo:
        raise ValueError(f"feature_range must be increasing, got {feature_range}.")

    mn = A.min(axis=0)
    mx = A.max(axis=0)
    rng = mx - mn
    flat = np.flatnonzero(rng == 0)
    if flat.size:
        raise ValueError(
            f"columns {flat.tolist()} are constant (range 0), so min-max scaling "
            f"divides by zero; drop them or use a standard scaler."
        )
    unit = (A - mn) / rng
    S = unit * (hi - lo) + lo

    return RichResult(
        title="Min-max scaling",
        summary_lines=[("Columns", int(A.shape[1])), ("Range", (lo, hi))],
        payload={
            "scaled": S.ravel().tolist() if vector else S.tolist(),
            "data_min": mn.tolist(),
            "data_max": mx.tolist(),
            "data_range": rng.tolist(),
            "scale": ((hi - lo) / rng).tolist(),
            "feature_range": (lo, hi),
            "estimate": S.ravel().tolist() if vector else S.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmms: x_scaled = (x - min)/(max - min) per column; constant column raises"
