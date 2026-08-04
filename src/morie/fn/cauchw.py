# morie.fn -- slice s05 (rootcoder007/morie)
"""Cauchy (Lorentzian) weight function for iteratively reweighted least squares.

Holland, P. W. and Welsch, R. E. (1977), "Robust regression using
iteratively reweighted least-squares", *Communications in Statistics -
Theory and Methods* 6(9), 813-827, doi:10.1080/03610927708827533.  The
bibliographic record is verified against Crossref.

CITATION LIMIT, stated rather than papered over.  The article is
closed access: Taylor and Francis returns 403 to every fetch, OpenAlex
and Semantic Scholar report no open-access location, and no preprint
or technical-report version was found.  No page or equation number is
therefore attributed to it.  The weight function below is the one this
module was specified with,

    w(r) = 1 / (1 + (r/c)^2),

and the companions follow from it by the standard M-estimation
relations rather than by quotation:

    psi(r) = r w(r) = r / (1 + (r/c)^2)
    rho(r) = (c^2 / 2) log(1 + (r/c)^2),   d rho / dr = psi.

The rho is the log-density of a Cauchy up to constants, which is where
the name comes from and why the function behaves as it does.  Unlike
Huber, the weight starts falling at r = 0 and never reaches zero:
every observation keeps some influence, however far out it lies, so
the estimator redescends but does not reject.  psi attains its maximum
at r = c and decreases thereafter, so a residual twice as large as c
carries LESS weight in absolute terms than one at c -- the property
Huber lacks and the bisquare takes to the extreme of exact rejection.

The default tuning constant c = 2.3849 is the value conventionally
quoted for about 95% asymptotic efficiency at the Gaussian; the module
does not assert it, and the accompanying anchor derives the efficiency
(E psi')^2 / E psi^2 by quadrature instead of trusting the number.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["cauchy_weight"]


def cauchy_weight(y, c=2.3849):
    """Cauchy weights, psi and rho for a vector of residuals.

    Parameters
    ----------
    y : array-like
        Residuals, already divided by a scale estimate if one is used.
    c : float
        Positive tuning constant.  Default 2.3849.

    Returns
    -------
    RichResult
        keys: ``estimate`` (the weights), ``weights``, ``psi``, ``rho``,
        ``objective`` (sum of rho), ``c``, ``n``, ``method``.

    References
    ----------
    Holland, P. W. and Welsch, R. E. (1977), *Communications in
    Statistics - Theory and Methods* 6(9):813-827,
    doi:10.1080/03610927708827533.
    """
    rv = core.vec(y)
    cc = float(c)
    if not (cc > 0.0) or math.isinf(cc):
        raise ValueError("cauchy_weight: c must be a positive finite number")
    n = len(rv)
    if n == 0:
        raise ValueError("cauchy_weight: y is empty")
    w = []
    psi = []
    rho = []
    for r in rv:
        u = r / cc
        d = 1.0 + u * u
        w.append(1.0 / d)
        psi.append(r / d)
        rho.append(0.5 * cc * cc * math.log(d))
    return RichResult(payload={
        "estimate": w, "weights": w, "psi": psi, "rho": rho,
        "objective": sum(rho), "c": cc, "n": int(n),
        "method": "Holland-Welsch (1977) Cauchy weight w(r) = 1/(1 + (r/c)^2)"})


def cheatsheet():
    return ("cauchw: downweights from r = 0 and never to zero -- redescends "
            "without rejecting; psi peaks at r = c")


# compact alias per ledger/NAMING.md
cauchyweight = cauchy_weight
