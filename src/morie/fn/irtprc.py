# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Masters' partial credit model.

Masters (1982), "A Rasch model for partial credit scoring",
Psychometrika 47(2):149-174, doi:10.1007/BF02296272, equation (7):

    P(X = k | theta) = exp( sum_{v=1}^{k} (theta - delta_v) )
                       / sum_{c=0}^{m} exp( sum_{v=1}^{c} (theta - delta_v) ),

with the empty sum at k = 0 equal to zero.  This is the Rasch member of
the partial-credit family: the slope is fixed at one, so the raw score
is a sufficient statistic for theta, which is what distinguishes it from
Muraki's GPCM.  The category probabilities are therefore obtained from
the GPCM kernel already on the shelf evaluated at a = 1 with the step
vector prefixed by a zero, rather than by writing a second softmax.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult
from .gpcm import _gpcm_probs

__all__ = ["partial_credit"]


def partial_credit(y, theta, delta_j):
    """Partial credit category probabilities and the observed likelihood.

    Parameters
    ----------
    y : array-like of int
        Observed categories, 0-based (0 .. m).
    theta : array-like
        Person abilities, same length as y.
    delta_j : array-like
        Step difficulties delta_1 .. delta_m; m + 1 categories result.
    """
    ys = [int(v) for v in core.vec(y)]
    th = core.vec(theta)
    dl = core.vec(delta_j)
    if len(ys) == 0:
        raise ValueError("partial_credit: y is empty")
    if len(th) != len(ys):
        raise ValueError("partial_credit: y and theta have different lengths")
    if len(dl) < 1:
        raise ValueError("partial_credit: delta_j needs at least one step")
    ncat = len(dl) + 1
    for v in ys:
        if v < 0 or v >= ncat:
            raise ValueError("partial_credit: y outside the category range")
    b = [0.0] + list(dl)
    pobs = []
    ll = 0.0
    for i in range(len(ys)):
        pr = _gpcm_probs(th[i], 1.0, b)
        pobs.append(pr[ys[i]])
        ll += math.log(pr[ys[i]])
    first = _gpcm_probs(th[0], 1.0, b)
    return RichResult(
        title="Partial credit model (Masters)",
        summary_lines=[("categories", ncat), ("loglik", ll)],
        payload={
            "estimate": sum(pobs) / len(pobs),
            "p_observed": pobs,
            "probs_first": first,
            "loglik": ll,
            "categories": ncat,
            "n": len(ys),
            "method": "P(X=k) = exp(sum_{v<=k}(theta - delta_v)) / normaliser, Masters (1982) eq. (7)",
        },
    )


def cheatsheet():
    return "irtprc: Partial credit model (Masters)"


# compact alias per ledger/NAMING.md
partialcredit = partial_credit
