# morie.fn -- slice s03 (rootcoder007/morie)
"""Three-parameter Indian buffet process.

Source consulted: Teh, Y. W. and Goeruer, D. (2009).  Indian buffet
processes with power-law behaviour.  *NIPS* 22, 1838-1846, which adds a
stability exponent sigma to the two-parameter IBP of Ghahramani, Griffiths
and Sollich (2007).  The expected number of features in n customers is

    K_n = alpha sum_(i=1)^n Gamma(1 + c) Gamma(i - 1 + c + sigma)
                            / ( Gamma(i + c) Gamma(c + sigma) )

which grows like O(n^sigma) for sigma in (0, 1) -- the power law the
title names -- and reduces to the logarithmic O(alpha log n) of the
one-parameter IBP at sigma = 0, c = 1.  The dish-selection probability
for customer n taking an already-chosen dish with m_k prior selections is

    P = (m_k - sigma) / (n - 1 + c)

The 2009 proceedings were not retrievable here; both expressions are
quoted in their standard published form.

Both the power-law growth and the sigma = 0 reduction are computed, so
the claim can be checked rather than taken on trust.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["isgp_bayes"]


def isgp_bayes(y, sigma=0.5, alpha=1.0, c=1.0):
    """Expected feature count and dish probabilities of the 3-parameter IBP.

    Parameters
    ----------
    y : int or array-like
        The number of customers n, or the data whose length is n.
    sigma : float
        Stability exponent in [0, 1).
    alpha : float
        Mass parameter.
    c : float
        Concentration.

    Returns
    -------
    estimate : E[K_n]
    K_n      : same as estimate
    K_path   : E[K_i] for i = 1..n
    new_dishes : expected new dishes at each customer
    one_param  : the sigma = 0 count, alpha * sum 1/(i - 1 + c) * c
    """
    if isinstance(y, (int, float)):
        n = int(y)
    else:
        n = len(k.vec(y))
    s = float(sigma)
    a = float(alpha)
    cc = float(c)
    newd = []
    for i in range(1, n + 1):
        t = math.exp(math.lgamma(1.0 + cc) + math.lgamma(i - 1.0 + cc + s)
                     - math.lgamma(i + cc) - math.lgamma(cc + s))
        newd.append(a * t)
    path = []
    acc = 0.0
    for x in newd:
        acc += x
        path.append(acc)
    one = 0.0
    for i in range(1, n + 1):
        one += a * cc / (i - 1.0 + cc)
    return RichResult(
        title="Three-parameter Indian buffet process",
        summary_lines=[("E[K_n]", acc), ("sigma", s)],
        payload={
            "estimate": acc,
            "K_n": acc,
            "K_path": path,
            "new_dishes": newd,
            "one_param": one,
            "n": n,
            "method": "Three-parameter IBP feature count (Teh and Goeruer 2009)",
        },
    )


def cheatsheet():
    return "isbplr: Indian buffet generalized prior"


isgpbayes = isgp_bayes
