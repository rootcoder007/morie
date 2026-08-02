# morie.fn -- function file (rootcoder007/morie)
"""TMLE for quantile treatment effects."""

from . import _array_core as np

from ._richresult import RichResult
from ._tmle import tmle_ate

__all__ = ["tmle_quantile"]


def tmle_quantile(y, D, X, quantile=0.5, n_grid=60, trunc=0.01):
    r"""Quantile treatment effect by TMLE on the counterfactual CDFs.

    A quantile is the inverse of a CDF, and each CDF value is itself a
    mean of an indicator, so

    .. math:: \hat F_a(t) = \widehat{E}\big[
              \mathbb{1}\{Y \le t\} \mid do(A=a)\big]

    is estimated by running the TMLE machinery on the indicator
    outcome at every grid point t. Inverting the two monotonised
    curves gives :math:`\hat q_1(\tau)` and :math:`\hat q_0(\tau)`,
    and the QTE is their difference. Monotonisation matters: the
    pointwise TMLE curves need not be monotone, so they are
    cumulative-maximum'd before inversion.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    quantile : float in (0, 1), default 0.5
    n_grid : int, default 60
        Grid points spanning the outcome range.
    trunc : float, default 0.01
        Propensity truncation.

    Returns
    -------
    RichResult
        keys: ``qte``, ``q1``, ``q0``, ``quantile``, ``grid``,
        ``f1``, ``f0`` (the monotonised counterfactual CDFs), ``n``,
        ``method``.

    References
    ----------
    Diaz, I. (2017). Efficient estimation of quantiles in missing data
    models. *Journal of Statistical Planning and Inference*, 190,
    39-51.

    van der Laan, M. J. & Rose, S. (2011). *Targeted Learning*.
    Springer.
    """
    y = np.asarray(y, dtype=float).ravel()
    q = float(quantile)
    if not 0 < q < 1:
        raise ValueError(f"quantile must lie in (0, 1), got {q}.")
    k = int(n_grid)
    if k < 5:
        raise ValueError(f"n_grid must be at least 5, got {k}.")

    grid = np.quantile(y, np.linspace(0.02, 0.98, k))
    f1, f0 = np.empty(k), np.empty(k)
    for i, t in enumerate(grid):
        ind = (y <= t).astype(float)
        if ind.min() == ind.max():  # degenerate grid point
            f1[i] = f0[i] = float(ind[0])
            continue
        out = tmle_ate(ind, D, X, trunc=trunc, scale_outcome=False)
        f1[i], f0[i] = out["ey1"], out["ey0"]

    f1 = np.clip(np.maximum.accumulate(f1), 0.0, 1.0)
    f0 = np.clip(np.maximum.accumulate(f0), 0.0, 1.0)

    def invert(F):
        idx = np.searchsorted(F, q)
        return float(grid[min(idx, k - 1)])

    q1, q0 = invert(f1), invert(f0)
    return RichResult(
        payload={
            "qte": q1 - q0,
            "q1": q1,
            "q0": q0,
            "quantile": q,
            "grid": grid,
            "f1": f1,
            "f0": f0,
            "n": int(y.size),
            "method": "TMLE quantile treatment effect (indicator TMLE + CDF inversion)",
        }
    )


def cheatsheet():
    return "tmlqct: TMLE each F_a(t) on 1{Y<=t}, monotonise, invert at tau"
