# morie.fn -- tail3 batch (rootcoder007/morie)
"""Catalogue coverage of a recommender.

Source consulted: Herlocker, J.L., Konstan, J.A., Terveen, L.G. & Riedl, J.T.
(2004). Evaluating collaborative filtering recommender systems.  *ACM
Transactions on Information Systems* 22(1), 5-53, section on coverage.  The
catalogue (or prediction) coverage of a recommender is the proportion of the
item catalogue that it actually recommends,

    coverage = |unique items recommended| / |catalogue|

which the paper pairs with accuracy because a recommender can buy accuracy by
only ever recommending the same few popular items.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["catalog_coverage"]


def catalog_coverage(recommendations, catalog):
    """Proportion of the catalogue that appears in the recommendation lists.

    Parameters
    ----------
    recommendations : sequence of sequences
        One recommendation list per user.
    catalog : sequence
        All items available to recommend.

    Returns
    -------
    RichResult
        estimate (coverage), covered, ncatalog, nlists, meanlen, n, method.

    References
    ----------
    Herlocker, Konstan, Terveen & Riedl (2004), ACM TOIS 22(1), 5-53.
    """
    cat = [float(v) for v in np.atleast_1d(np.asarray(catalog, dtype=float)).ravel()]
    catset = set(cat)
    seen = set()
    nlists = 0
    total = 0
    for lst in recommendations:
        nlists += 1
        items = [float(v) for v in np.atleast_1d(np.asarray(lst, dtype=float)).ravel()]
        total += len(items)
        for it in items:
            if it in catset:
                seen.add(it)
    ncat = len(catset)
    cov = float(len(seen)) / ncat if ncat > 0 else float("nan")
    return RichResult(
        payload={
            "estimate": cov,
            "coverage": cov,
            "covered": int(len(seen)),
            "ncatalog": int(ncat),
            "nlists": int(nlists),
            "meanlen": float(total) / nlists if nlists > 0 else float("nan"),
            "n": int(nlists),
            "method": "Catalogue coverage (Herlocker et al. 2004)",
        }
    )


# CANONICAL TEST
# >>> # two lists covering 3 of 6 catalogue items
# >>> r = catalog_coverage([[1, 2], [2, 3]], [1, 2, 3, 4, 5, 6])
# >>> assert abs(r["estimate"] - 0.5) < 1e-12
# >>> assert r["covered"] == 3


def cheatsheet():
    return "covLst(recommendations, catalog): catalogue coverage."
