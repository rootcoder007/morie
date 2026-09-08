# morie.fn -- slice s03 (rootcoder007/morie)
"""Intra-list diversity of a recommendation list.

Source consulted: Ziegler, C.-N., McNee, S. M., Konstan, J. A. and
Lausen, G. (2005).  Improving recommendation lists through topic
diversification.  *WWW* 14, 22-32, which defines the intra-list
similarity of a list P_w as

    ILS(P_w) = sum_(b_i in P_w) sum_(b_j in P_w, b_j != b_i) c_0(b_i, b_j) / 2

so that the *diversity* is the complementary average
mean_(i < j) (1 - s(i, j)) over the C(k, 2) unordered pairs, which is
the module's own formula line.  The 2005 WWW proceedings were not
retrievable here; the definition is quoted in its standard published
form.  The count of distinct pairs is reported so that the average is
interpretable when the list is short.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["diversity"]


def diversity(list, sim_matrix=None):
    """Average pairwise dissimilarity over a recommended list.

    Parameters
    ----------
    list : array-like
        Indices of the recommended items.
    sim_matrix : 2-D array-like
        Item-item similarity, indexed by those indices.

    Returns
    -------
    RichResult with payload:
        estimate : mean_(i<j) (1 - s(i, j))
        ils      : the intra-list similarity sum of Ziegler et al.
        n_pairs  : C(k, 2)
        min_pair_sim, max_pair_sim
    """
    items = [int(x) for x in list]
    S = k.mat(sim_matrix)
    kk = len(items)
    tot = 0.0
    ils = 0.0
    np_ = 0
    lo = float("inf")
    hi = float("-inf")
    for a in range(kk):
        for b in range(a + 1, kk):
            s = S[items[a]][items[b]]
            tot += 1.0 - s
            ils += s
            np_ += 1
            if s < lo:
                lo = s
            if s > hi:
                hi = s
    return RichResult(
        title="Intra-list diversity",
        summary_lines=[("diversity", tot / np_ if np_ else float("nan"))],
        payload={
            "estimate": tot / np_ if np_ else float("nan"),
            "ils": ils,
            "n_pairs": np_,
            "min_pair_sim": lo if np_ else float("nan"),
            "max_pair_sim": hi if np_ else float("nan"),
            "method": "Intra-list diversity, the complement of Ziegler et al. (2005) ILS",
        },
    )


def cheatsheet():
    return "divLst: Intra-list diversity"
