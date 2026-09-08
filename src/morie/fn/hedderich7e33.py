# morie.fn -- slice k04 (rootcoder007/morie)
"""Anderson-Darling goodness-of-fit statistic -- Hedderich eq (7.33).

Source READ FROM THE CORPUS PDF: Hedderich, Sachs and Reynarowych,
*Applied Statistics: Methods Using R*, section 7.2.8 "Anderson-Darling
Test", equation (7.33), citing Anderson and Darling (1952) and Stephens
(1986b).  The corpus text layer renders (7.33) as::

    A2 = -N - S    with
    S = sum_{i=1}^{N} (2i - 1)/N [logF (Yi ) + log(1 - YN +1-i )]n sigma 2

Two extraction defects in that line are corrected here against the
surrounding prose and against Anderson and Darling (1952):

  * the trailing ``n sigma 2`` is spillover from the neighbouring
    column and is not part of the statistic;
  * ``1 - YN+1-i`` is ``1 - F(Y_{N+1-i})`` -- the second logarithm takes
    the same probability-integral transform as the first, not the raw
    order statistic.

The statistic actually computed, with ``u_i = F(Y_(i))`` the sorted
probability-integral transform, is therefore

    A2 = -N - (1/N) sum_{i=1}^{N} (2i - 1) [ log u_i + log(1 - u_{N+1-i}) ]

which is the standard published form.  The book notes that critical
values must be derived separately for each distribution model, so no
p-value is returned: the statistic alone is the honest output.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["hedderich_chapter_7_equation_33"]

# Smallest and largest PIT value tolerated before log() blows up.  A u of
# exactly 0 or 1 means the fitted model gives an observation probability
# zero, which makes A2 infinite; that is the correct answer, so it is
# reported as such rather than being clipped into a finite lie.
_EPS = 0.0


def ad_statistic(u):
    """Anderson-Darling A^2 from probability-integral transforms.

    Parameters
    ----------
    u : array-like
        The PIT values ``F(y_i)`` under the fitted model, in any order.
        They are sorted here.

    Returns
    -------
    float
        ``A^2``; ``inf`` if any value is 0 or 1.
    """
    u = np.asarray(u, dtype=float).ravel()
    n = int(u.size)
    if n < 2:
        raise ValueError("need at least 2 observations")
    us = np.sort(u)
    if float(us[0]) <= _EPS or float(us[-1]) >= 1.0 - _EPS:
        return float("inf")
    s = 0.0
    for i in range(1, n + 1):
        s += (2 * i - 1) * (math.log(float(us[i - 1])) + math.log(1.0 - float(us[n - i])))
    return -n - s / n


def hedderich_chapter_7_equation_33(y, cdf):
    """Anderson-Darling A^2 for a fully specified distribution.

    Parameters
    ----------
    y : array-like
        Sample.
    cdf : callable
        The hypothesised distribution function ``F``, applied
        elementwise.  It must be fully specified: the book stresses that
        Anderson-Darling critical values depend on the model, so a cdf
        with parameters estimated from ``y`` changes the null
        distribution and is the caller's responsibility to account for.

    Returns
    -------
    RichResult
        keys: ``statistic`` (A^2), ``u`` (sorted PIT values), ``n``,
        ``method``.
    """
    y = np.asarray(y, dtype=float).ravel()
    u = np.array([float(cdf(float(v))) for v in y], dtype=float)
    return RichResult(
        payload={
            "statistic": ad_statistic(u),
            "u": np.sort(u),
            "n": int(y.size),
            "method": "Anderson-Darling A^2 (Hedderich eq. 7.33; Anderson and Darling 1952)",
        }
    )


def cheatsheet():
    return "hedderich7e33: Anderson-Darling A^2 statistic (eq. 7.33)"
