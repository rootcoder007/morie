# morie.fn -- function file (rootcoder007/morie)
"""Infinitesimal-jackknife variance for causal-forest predictions."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_forest_variance"]


def causal_forest_variance(forest, X_test=None, bias_correct=True):
    r"""Pointwise variance of a causal forest by the infinitesimal jackknife.

    Wager and Athey (2018), Theorem 9. Each training observation's
    influence on the prediction is read off the covariance between its
    inclusion in a tree's subsample and that tree's prediction:

    .. math:: \hat V_{IJ}(x) = \frac{n-1}{n}
              \left(\frac{n}{n-s}\right)^{2}
              \sum_{i=1}^{n}
              \widehat{\mathrm{Cov}}\big(N_{ib},\,
              \hat\tau_b(x)\big)^2,

    with :math:`N_{ib}` the indicator that observation :math:`i` was in
    tree :math:`b`'s subsample and :math:`s` the subsample size.

    The finite-:math:`B` correction is not optional. With a limited
    number of trees the raw sum is inflated by Monte-Carlo noise in the
    covariances themselves, and the bias does not vanish as :math:`n`
    grows -- only as :math:`B` does. Left uncorrected the intervals are
    systematically too wide, which reads as conservative but is simply
    wrong. ``bias_raw`` reports how large that correction was, and
    ``correction_share`` how much of the raw estimate it removed; a
    share near or above one means the forest has too few trees for the
    variance to be estimated at all, and the result says so.

    Parameters
    ----------
    forest : CausalForest or RichResult
        A fitted forest, or the result of
        :func:`~morie.fn.cfst.causal_forest` (its ``forest`` entry is
        used).
    X_test : array-like, optional
        Points at which to evaluate. The training rows by default.
    bias_correct : bool
        Apply the finite-tree correction.

    Returns
    -------
    RichResult
        ``variance``, ``se``, ``ci_lower``, ``ci_upper``,
        ``predictions``, ``variance_raw``, ``bias_raw``,
        ``correction_share``, ``n_trees``, ``reliable``.

    References
    ----------
    Wager and Athey (2018), *JASA* 113:1228-1242, Theorem 9.
    Efron (2014), *JASA* 109:991-1007 (the infinitesimal jackknife).
    Wager, Hastie and Efron (2014), *JMLR* 15:1625-1651 (the
    finite-:math:`B` bias correction).

    Examples
    --------
    >>> import numpy as np
    >>> from morie.fn.cfst import causal_forest
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 2))
    >>> T = (rng.uniform(size=600) < 0.5).astype(float)
    >>> Y = T * X[:, 0] + rng.normal(scale=0.3, size=600)
    >>> f = causal_forest(Y, T, X, n_trees=120, seed=2)
    >>> v = causal_forest_variance(f, X[:5])
    >>> bool(np.all(v["se"] > 0))
    True
    """
    if hasattr(forest, "get") and not hasattr(forest, "trees_"):
        forest = forest.get("forest")
    if forest is None or not getattr(forest, "trees_", None):
        raise ValueError(
            "pass a fitted causal forest, or the result of causal_forest "
            "(whose 'forest' entry holds one)."
        )
    B = len(forest.trees_)
    if B < 2:
        raise ValueError(
            "the infinitesimal jackknife needs at least 2 trees, got %d." % B
        )
    n = forest._n
    N = np.array(forest.in_bag_, dtype=float)          # (B, n)
    s = float(N.sum(axis=1).mean())
    if s >= n:
        raise ValueError(
            "every tree used the whole sample, so the inclusion indicators "
            "carry no information; refit with subsample < 1."
        )

    Xq = forest._X if X_test is None else np.asarray(X_test, dtype=float)
    if Xq.ndim == 1:
        Xq = Xq[:, None]
    # per-tree predictions at each query point
    P = np.array([[forest._walk(t, row) for row in Xq]
                  for t in forest.trees_])             # (B, m)
    Pc = P - P.mean(axis=0, keepdims=True)
    Nc = N - N.mean(axis=0, keepdims=True)
    cov = (Nc.T @ Pc) / B                              # (n, m)
    raw = np.sum(cov**2, axis=0)
    scale = (n - 1.0) / n * (n / (n - s)) ** 2
    raw = scale * raw

    # Finite-B correction. Each covariance is itself an average over B
    # trees, so it carries Monte-Carlo noise of size
    # Var(N_ib) * Var(tau_b) / B, and squaring it leaves that noise in
    # the sum. Summing over the n observations and using
    # Var(N_ib) = (s/n)(1 - s/n) for sampling without replacement gives
    # s (1 - s/n) / B times the per-tree prediction variance.
    var_tree = P.var(axis=0, ddof=1)
    bias = scale * (s * (1.0 - s / n) / B) * var_tree
    if bias_correct:
        var = np.maximum(raw - bias, 0.0)
    else:
        var = raw
    share = np.where(raw > 0, bias / raw, np.nan)

    se = np.sqrt(var)
    pred = P.mean(axis=0)
    z = 1.959963984540054
    reliable = bool(np.nanmedian(share) < 0.9) if share.size else False
    return RichResult(
        payload={
            "estimate": pred,
            "predictions": pred,
            "variance": var,
            "se": se,
            "ci_lower": pred - z * se,
            "ci_upper": pred + z * se,
            "variance_raw": raw,
            "bias_raw": bias,
            "correction_share": share,
            "bias_corrected": bool(bias_correct),
            "correction_note": (
                "the raw infinitesimal jackknife is inflated by Monte-Carlo "
                "noise in its own covariances; that bias shrinks with the "
                "number of TREES, not the sample size, so it must be "
                "subtracted rather than out-grown"
            ),
            "reliable": reliable,
            "reliability_note": (
                None if reliable else
                "the correction removes most of the raw variance, which "
                "means %d trees is too few to estimate it; grow more before "
                "reading these intervals" % B
            ),
            "n_trees": B,
            "subsample_size": s,
            "n_train": int(n),
            "n_query": int(Xq.shape[0]),
            "method": "Infinitesimal-jackknife variance for a causal forest",
        }
    )


def cheatsheet():
    return (
        "crfvar: pointwise causal-forest variance by the infinitesimal "
        "jackknife with the finite-tree bias correction, and a flag when "
        "there are too few trees to trust it"
    )
