# morie.fn -- function file (rootcoder007/morie)
"""Impurity-based feature importance for tree ensembles."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["feature_importance_trees", "geron_feature_importance"]


def feature_importance_trees(impurity_decrease, n_features=None,
                             normalize=True, feature_names=None):
    r"""Average impurity decrease per feature across an ensemble.

    .. math::
       I_f = \frac{1}{B}\sum_{b=1}^{B}
             \sum_{\text{nodes } t \in T_b \,:\, v(t)=f}
             \frac{n_t}{n}\,\Delta i(t)

    the weighted impurity decrease attributable to feature :math:`f`,
    averaged over trees.

    This measure has two failure modes that matter more than the
    ranking it produces. It is BIASED TOWARD HIGH-CARDINALITY features:
    a continuous variable or one with many categories offers more split
    points, so it wins more splits by chance alone. And with correlated
    predictors it SPLITS the credit arbitrarily -- two copies of the
    same variable each receive about half the importance one of them
    would get alone, which reads as both being unimportant.

    ``concentration`` (the Gini coefficient of the importances) and
    ``effective_features`` (the inverse Simpson index) describe how the
    total is spread, so a diffuse ranking is visible rather than being
    read off as a list of top features. Permutation importance on held
    out data avoids both biases and is the honest alternative when the
    ranking carries weight.

    Parameters
    ----------
    impurity_decrease : array-like, shape (B, p) or (p,)
        Per-tree, per-feature weighted impurity decrease.
    n_features : int, optional
        Checked against the data's width.
    normalize : bool
        Scale to sum to one.
    feature_names : sequence, optional

    Returns
    -------
    RichResult
        ``importance``, ``ranking``, ``concentration``,
        ``effective_features``, ``stability`` (across trees).

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 7,
    feature importance. Breiman (2001). Strobl et al. (2007), *BMC
    Bioinformatics* 8:25, on the cardinality bias.

    Examples
    --------
    >>> out = feature_importance_trees([[0.6, 0.4], [0.8, 0.2]])
    >>> [round(float(v), 3) for v in out["importance"]]
    [0.7, 0.3]
    """
    A = np.atleast_2d(np.asarray(impurity_decrease, dtype=float))
    if A.ndim != 2:
        raise ValueError("impurity_decrease must be 1- or 2-dimensional.")
    B, p = A.shape
    if n_features is not None and int(n_features) != p:
        raise ValueError(
            "n_features says %d but the data has %d columns."
            % (int(n_features), p)
        )
    if np.any(A < -1e-12):
        raise ValueError("impurity decrease cannot be negative.")
    imp = A.mean(axis=0)
    total = float(imp.sum())
    if normalize and total > 0:
        imp = imp / total
    order = np.argsort(imp)[::-1]
    names = (list(feature_names) if feature_names is not None
             else ["x%d" % j for j in range(p)])
    if len(names) != p:
        raise ValueError(
            "feature_names has %d entries for %d features." % (len(names), p)
        )
    s = imp.sum()
    share = imp / s if s > 0 else np.full(p, np.nan)
    # Gini of the shares, and the inverse Simpson index
    srt = np.sort(share)
    idx = np.arange(1, p + 1)
    gini = float((2 * np.sum(idx * srt) / (p * np.sum(srt)) - (p + 1) / p)) \
        if s > 0 and p > 1 else 0.0
    eff = float(1.0 / np.sum(share ** 2)) if s > 0 else np.nan
    stab = (float(np.mean(np.std(A, axis=0) / np.maximum(A.mean(axis=0), 1e-12)))
            if B > 1 else np.nan)
    return RichResult(
        payload={
            "estimate": imp,
            "importance": imp,
            "ranking": [names[j] for j in order],
            "order": order,
            "feature_names": names,
            "concentration": gini,
            "effective_features": eff,
            "effective_note": (
                "inverse Simpson index of the importance shares: how many "
                "features the total is effectively spread across"
            ),
            "stability": stab,
            "bias_note": (
                "impurity importance favours high-cardinality features, "
                "which offer more split points by chance, and divides credit "
                "arbitrarily between correlated ones; permutation importance "
                "on held-out data has neither problem"
            ),
            "normalized": bool(normalize),
            "n_trees": int(B),
            "n_features": int(p),
            "method": "Impurity-based feature importance",
        }
    )


def cheatsheet():
    return (
        "hmfim: mean impurity decrease per feature, with the cardinality and "
        "correlation biases stated and the spread quantified"
    )


#: Catalogue alias for :func:`feature_importance_trees`.
geron_feature_importance = feature_importance_trees
