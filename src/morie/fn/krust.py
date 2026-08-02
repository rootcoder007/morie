# morie.fn -- function file (rootcoder007/morie)
"""Kruskal stress-1 badness of fit for MDS solutions."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kruskal_stress"]


def kruskal_stress(D_observed, D_config):
    r"""Kruskal's stress-1.

    .. math:: S_1 = \sqrt{\frac{\sum_{i<j} (\hat d_{ij} - d_{ij})^2}
              {\sum_{i<j} d_{ij}^2}},

    where :math:`d_{ij}` are the configuration distances and
    :math:`\hat d_{ij}` the target dissimilarities (or disparities in
    the nonmetric loop). Kruskal's verbal scale: 0.20 poor, 0.10 fair,
    0.05 good, 0.025 excellent, 0 perfect.

    Parameters
    ----------
    D_observed : array-like, shape (n, n)
        Target dissimilarities/disparities.
    D_config : array-like, shape (n, n)
        Distances of the fitted configuration.

    Returns
    -------
    RichResult
        keys: ``stress`` , ``verbal`` (Kruskal's label), ``n``,
        ``method``.

    References
    ----------
    Kruskal, J. B. (1964). Multidimensional scaling by optimizing
    goodness of fit to a nonmetric hypothesis. *Psychometrika*, 29(1),
    1-27. (stress-1 and the verbal scale)
    """
    Do = np.asarray(D_observed, dtype=float)
    Dc = np.asarray(D_config, dtype=float)
    if Do.shape != Dc.shape or Do.ndim != 2 or Do.shape[0] != Do.shape[1]:
        raise ValueError("D_observed and D_config must be square matrices of the same shape.")
    n = Do.shape[0]
    iu = np.triu_indices(n, k=1)
    num = float(((Do[iu] - Dc[iu]) ** 2).sum())
    den = float((Dc[iu] ** 2).sum())
    if den <= 0:
        raise ValueError("configuration distances are all zero; stress undefined.")
    s = float(np.sqrt(num / den))
    verbal = ("perfect" if s == 0 else "excellent" if s <= 0.025 else "good" if s <= 0.05
              else "fair" if s <= 0.10 else "poor" if s <= 0.20 else "unacceptable")

    return RichResult(
        payload={
            "stress": s,
            "verbal": verbal,
            "n": int(n),
            "method": "Kruskal stress-1",
        }
    )


def cheatsheet():
    return "krust: S1 = sqrt(sum(dhat-d)^2 / sum d^2); 0.05 good, 0.20 poor (Kruskal 1964)"
