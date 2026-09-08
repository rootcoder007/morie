# morie.fn -- function file (rootcoder007/morie)
"""Quantile-balanced causal forest for distributional treatment effects."""

from . import _array_core as np

from ._cforest import CausalForest
from ._richresult import RichResult

__all__ = ["quantile_balanced_cf"]


def quantile_balanced_cf(y, D, X, quantile=0.5, n_trees=200, min_leaf=15, seed=0):
    r"""Heterogeneous *quantile* treatment effects via an indicator forest.

    The trick is that a quantile contrast is a mean contrast of an
    indicator: for the pooled :math:`\tau`-quantile threshold
    :math:`q_\tau`,

    .. math:: P(Y \le q_\tau \mid D=1, X)
              - P(Y \le q_\tau \mid D=0, X)

    is estimated by running the honest causal forest on
    :math:`\mathbb{1}\{Y \le q_\tau\}`. A negative value at the median
    means treatment pushes mass above the threshold -- the sign
    convention is opposite to a location shift, so it is reported as
    ``shift_effect`` with the sign flipped as well.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment.
    X : array-like, shape (n, p) or (n,)
        Covariates.
    quantile : float in (0, 1), default 0.5
        Quantile level.
    n_trees, min_leaf, seed :
        Forest hyperparameters.

    Returns
    -------
    RichResult
        keys: ``cdf_effect`` (n,, the P(Y<=q) contrast),
        ``shift_effect`` (its negation, positive = treatment shifts
        the distribution up), ``threshold`` (q_tau), ``quantile``,
        ``ate_cdf``, ``n``, ``forest``, ``method``.

    References
    ----------
    Athey, S., Tibshirani, J. & Wager, S. (2019). Generalized random
    forests. *The Annals of Statistics*, 47(2), 1148-1178.
    (distributional / quantile-target forests)
    """
    y = np.asarray(y, dtype=float).ravel()
    q = float(quantile)
    if not 0 < q < 1:
        raise ValueError(f"quantile must lie in (0, 1), got {q}.")
    thr = float(np.quantile(y, q))
    ind = (y <= thr).astype(float)

    f = CausalForest(n_trees=n_trees, min_leaf=min_leaf, seed=seed)
    f.fit(X, ind, D)
    cdf_eff = f.predict()

    return RichResult(
        payload={
            "cdf_effect": cdf_eff,
            "shift_effect": -cdf_eff,
            "threshold": thr,
            "quantile": q,
            "ate_cdf": float(np.nanmean(cdf_eff)),
            "n": int(y.size),
            "forest": f,
            "method": "Quantile-balanced causal forest (indicator-outcome CDF contrast)",
        }
    )


def cheatsheet():
    return "qbcfgr: run the honest forest on 1{Y <= q_tau}; negate for the shift direction"
