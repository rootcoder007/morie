# morie.fn -- function file (rootcoder007/morie)
"""EAP ability estimate, its posterior sd, and test information."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["eapinfo", "eap_information"]


def eapinfo(items, x, D=1.0, prior_mean=0.0, prior_sd=1.0,
            lower=-4.0, upper=4.0, nqp=33):
    """EAP ability estimate with its posterior sd and Fisher information.

    Two different "standard errors" live here and they are not the
    same number: the posterior sd of the EAP shrinks towards the prior
    and stays finite even for an all-correct pattern, while
    1/sqrt(I(theta)) is the frequentist error of the maximum-likelihood
    estimate and diverges there.  Both are returned so the difference
    is visible rather than assumed away.

    The quadrature grid is FIXED (nqp points on [lower, upper],
    trapezoid rule), never adaptive, so the two language arms integrate
    the same points.

    ``items`` rows are (a, b, c, d): discrimination, difficulty,
    lower asymptote, upper asymptote -- the four-parameter
    parameterisation, with (a, b, 0, 1) giving the 2PL.

    Formula: e = exp(D a (theta - b));  P = c + (d - c) e/(1 + e);
             dP = D a e (d - c)/(1 + e)^2;  I_j = dP^2/(P(1 - P));
             EAP = int theta pi(theta) L(theta) dtheta
                   / int pi(theta) L(theta) dtheta

    Parameters
    ----------
    items : array-like, shape (J, 4)
        Item parameters (a, b, c, d).
    x : array-like
        Responses in {0, 1}, length J.
    D : float
        Scaling constant (1 for the logistic metric, 1.702 for normal).
    prior_mean, prior_sd : float
        Normal prior on theta.
    lower, upper : float
        Quadrature range.
    nqp : int
        Number of quadrature points.

    Returns
    -------
    RichResult
        ``estimate`` (EAP), ``se`` (posterior sd), ``information``
        (test information at the EAP), ``se_ml`` (1/sqrt(I)),
        ``item_information``, ``prob``, ``J``, ``nqp``.

    References
    ----------
    Item response function, derivative and information verified
    against the reference implementation in the CRAN package ``catR``
    3.17 (Magis & Raiche), functions ``Pi`` and ``Ii``:
    ``e <- exp(D * a * (th - b)); Pi <- c + (d - c) * e/(1 + e);
    dPi <- D * a * e * (d - c)/(1 + e)^2; Ii <- dP^2/(P*Q)``.  The EAP
    is that package's ``eapEst``, a ratio of two quadratures of
    prior times likelihood over a fixed grid.  ``catR`` implements the
    procedures of van der Linden & Pashley, Item selection and ability
    estimation in adaptive testing, in Elements of Adaptive Testing
    (2010), which this row cites; that chapter was NOT obtainable, so
    the package source is used as the reference implementation.
    """
    It = C.mat(items)
    J = len(It)
    if J < 1:
        raise ValueError("at least one item is required")
    if any(len(r) != 4 for r in It):
        raise ValueError("item rows must be (a, b, c, d)")
    x = C.vec(x)
    if len(x) != J:
        raise ValueError("one response per item is required")
    if any(v not in (0.0, 1.0) for v in x):
        raise ValueError("responses must be 0 or 1")
    D = float(D)
    ps = float(prior_sd)
    if ps <= 0:
        raise ValueError("the prior sd must be positive")
    nq = int(nqp)
    if nq < 3:
        raise ValueError("at least three quadrature points are required")
    h = (float(upper) - float(lower)) / (nq - 1)
    grid = [float(lower) + i * h for i in range(nq)]

    def probs(th):
        out = []
        for a, b, c, d in It:
            e = math.exp(D * a * (th - b))
            p = c + (d - c) * e / (1.0 + e)
            p = min(1.0 - 1e-10, max(1e-10, p))
            out.append(p)
        return out

    num = []
    den = []
    for th in grid:
        p = probs(th)
        L = 1.0
        for j in range(J):
            L *= p[j] if x[j] == 1.0 else (1.0 - p[j])
        pr = math.exp(-0.5 * ((th - float(prior_mean)) / ps) ** 2) / ps
        den.append(pr * L)
        num.append(th * pr * L)

    def trap(v):
        return h * (0.5 * v[0] + sum(v[1:-1]) + 0.5 * v[-1])

    d0 = trap(den)
    if d0 <= 0.0:
        raise ValueError("the posterior integrated to zero; check the grid")
    eap = trap(num) / d0
    sq = [grid[i] ** 2 * den[i] for i in range(nq)]
    var = trap(sq) / d0 - eap * eap
    p = probs(eap)
    info = []
    for j, (a, b, c, d) in enumerate(It):
        e = math.exp(D * a * (eap - b))
        dp = D * a * e * (d - c) / (1.0 + e) ** 2
        info.append(dp * dp / (p[j] * (1.0 - p[j])))
    tot = sum(info)
    return RichResult(payload={
        "estimate": eap, "se": math.sqrt(var) if var > 0 else 0.0,
        "information": tot,
        "se_ml": 1.0 / math.sqrt(tot) if tot > 0 else float("inf"),
        "item_information": info, "prob": p, "J": float(J),
        "nqp": float(nq),
        "method": "EAP with posterior sd and test information"})


eap_information = eapinfo


def cheatsheet():
    return "esatic: EAP on a fixed grid; posterior sd vs 1/sqrt(I) both returned"
