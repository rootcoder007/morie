# morie.fn -- function file (rootcoder007/morie)
"""Local Geary's c per location."""

from . import _tail1core as C
from ._richresult import RichResult

__all__ = ["localgeary", "local_gearys_c", "localgearysc"]


def localgeary(x, W, scale=True):
    """Local Geary's c per location.

    Where local Moran asks whether a location resembles its neighbours in sign, local Geary asks about squared difference, so it reacts to a location that is unlike its neighbours even when the global pattern is positive autocorrelation. x is standardised first, as the reference implementation does, using the sample standard deviation.


    Formula: c_i = sum_j w_ij (z_i - z_j)^2 with z the standardised x.
    With several variables (x of shape (n, k)) this is the multivariate
    local Geary of Anselin (2019), c_i = (1/k) sum_v sum_j w_ij (z_iv - z_jv)^2,
    as spdep::localC on a list of variables.

    Parameters
    ----------
    x : array-like
        Values at the n locations, shape (n,) or (n, k) with one variable
        per column.
    W : array-like, shape (n, n)
        Spatial weights.
    scale : bool
        Standardise x before computing c_i.

    Returns
    -------
    RichResult
        ``local``, ``global_c``, ``z`` (one list per variable when k > 1),
        ``n``, ``n_variables``.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> [round(c, 6) for c in localgeary([1.0, 2.0, 4.0, 8.0], W)["local"]]
    [0.104348, 0.521739, 2.086957, 1.669565]

    References
    ----------
    Anselin (2019), A Local Indicator of Multivariate Spatial
    Association: Extending Geary's c, Geographical Analysis 51:133-150.
    Paywalled; the univariate form c_i = sum_j w_ij (x_i - x_j)^2 and the
    standardise-first convention are as documented by spdep::localC, the
    reference implementation.
    """
    rows = [list(r) if isinstance(r, (list, tuple)) else r for r in (x.tolist() if hasattr(x, "tolist") else list(x))]
    multi = bool(rows) and isinstance(rows[0], list)
    cols = [list(c) for c in zip(*rows)] if multi else [rows]
    W = C.mat(W)
    zs = []
    for col in cols:
        xv = C.vec(col)
        n = len(xv)
        if scale:
            mu = sum(xv) / n
            s = C.sd(xv, 1)
            if s <= 0:
                raise ValueError("x has zero variance")
            zs.append([(v - mu) / s for v in xv])
        else:
            zs.append(list(xv))
    n = len(zs[0])
    k = len(zs)
    loc = [sum(W[i][j] * (z[i] - z[j]) ** 2 for z in zs for j in range(n)) / k for i in range(n)]
    s0 = sum(sum(row) for row in W)
    return RichResult(
        payload={
            "local": loc,
            "global_c": sum(loc) / (2.0 * s0) if s0 else float("nan"),
            "z": zs if multi else zs[0],
            "n": n,
            "n_variables": k,
            "method": "Local Geary's c" if k == 1 else "Multivariate local Geary's c",
        }
    )


local_gearys_c = localgeary
localgearysc = localgeary


def cheatsheet():
    return "gearyl: Local Geary's c per location."
