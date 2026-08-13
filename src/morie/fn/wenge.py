# morie.fn -- function file (rootcoder007/morie)
r"""Inverse-odds weighting for causal mediation.

Tchetgen Tchetgen & Shpitser (2012) give three representations of the
mediation functional -- what Pearl calls the mediation formula --

.. math:: \theta_0 = \iint E(Y \mid E{=}1, M{=}m, X{=}x)\,
          f_{M\mid E,X}(m \mid E{=}0, x)\, f_X(x)\, d\mu(m,x),

which is :math:`E(Y_{1,M_0})`, the outcome had everyone been exposed
but the mediator kept at the level it would have taken unexposed. All
three are implemented, because the paper's central structural point is
that they are *the same number* on a nonparametric model and diverge
only once parametric models are imposed:

``"ym"``
    Strategy 1, the plug-in: fit the outcome regression and the
    mediator density and integrate.
``"ye"``
    Strategy 2, :math:`P_n\!\left[\frac{I(E=0)}{f_{E|X}(0|X)}
    \hat E(Y \mid E{=}1, M, X)\right]`.
``"em"``
    Strategy 3, the inverse-odds form the module is named for,

    .. math:: P_n\!\left[Y\,
              \frac{I(E=1)}{f_{E|X}(E|X)}\,
              \frac{f_{M|E,X}(M \mid E{=}0, X)}
                   {f_{M|E,X}(M \mid E, X)}\right],

    which reweights the exposed by the odds of having been unexposed at
    their observed mediator value.

The paper states the equivalence directly: "if the estimated joint
distribution ... satisfies the positivity assumption ... then actually
theta_em = theta_ye = theta_ym", so on a saturated model the three
agree, are all efficient, and have a common influence function. The
anchor checks that they agree to machine precision on a discrete
saturated example -- and that they part company when the outcome model
is misspecified, which is the paper's reason for wanting more than one.

The natural direct and indirect effects follow, with
:math:`\delta_0 = E(Y_0)`:

.. math:: \mathrm{NDE} = \theta_0 - \delta_0, \qquad
          \mathrm{NIE} = E(Y_1) - \theta_0.

**Assumptions, stated because they are not checkable from the data.**
Consistency; sequential ignorability, which requires no exposure-induced
confounder of the mediator-outcome relation; and positivity of both
:math:`f_{E|X}` and :math:`f_{M|E,X}`. The positivity part *is*
checkable and is checked -- a zero density raises rather than producing
a weight of infinity that quietly becomes a number.

References
----------
Tchetgen Tchetgen, E. J. & Shpitser, I. (2012) "Semiparametric theory
for causal mediation analysis: efficiency bounds, multiple robustness
and sensitivity analysis", *The Annals of Statistics* 40(3), 1816-1845,
doi:10.1214/12-AOS990; arXiv:1210.4654. Equation (2) and the three
strategies of Sec. 3.

Imai, K., Keele, L. & Tingley, D. (2010) "A general approach to causal
mediation analysis", *Psychological Methods* 15(4), 309-334,
doi:10.1037/a0020761 -- the sequential ignorability condition the paper
adopts.

Pearl, J. (2001) "Direct and indirect effects", *Proceedings of the
Seventeenth Conference on Uncertainty in Artificial Intelligence*,
411-420 -- natural direct and indirect effects.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["weight_based_mediation", "mediation_functional"]

_STRATEGIES = ("em", "ye", "ym", "all")


def _cell_key(row):
    return tuple(round(float(v), 12) for v in row)


def mediation_functional(Y, E, M, X, strategy="em", saturated=True,
                         ridge=1e-8):
    r"""theta_0 = E(Y_{1, M_0}) by one of the paper's three strategies.

    `saturated=True` estimates every conditional nonparametrically by
    cell means over the distinct values of (E, M, X), which is the
    setting in which the three strategies provably coincide. Set it
    False to use logistic and linear working models instead, which is
    where they can disagree -- and that disagreement is diagnostic, not
    noise.
    """
    if strategy not in _STRATEGIES:
        raise ValueError("mediation_functional: strategy must be one of "
                         "%r, got %r" % (_STRATEGIES, strategy))
    yv, ev = k.vec(Y), k.vec(E)
    Mm = k.mat(M) if M is not None else [[0.0]] * len(yv)
    Xm = k.mat(X) if X is not None else [[0.0]] * len(yv)
    n = len(yv)
    for name, arr in (("E", ev), ("M", Mm), ("X", Xm)):
        if len(arr) != n:
            raise ValueError("mediation_functional: Y has %d rows but %s "
                             "has %d" % (n, name, len(arr)))
    if any(v not in (0.0, 1.0) for v in ev):
        raise ValueError("mediation_functional: E must be binary 0/1; the "
                         "mediation formula of eq. (2) is defined for a "
                         "binary exposure")

    if saturated:
        fe1, fm = _saturated_models(ev, Mm, Xm)
        ey = _saturated_outcome(yv, ev, Mm, Xm)
    else:
        fe1, fm, ey = _parametric_models(yv, ev, Mm, Xm, ridge)

    for i in range(n):
        p = fe1(i)
        if p <= 0.0 or p >= 1.0:
            raise ValueError(
                "mediation_functional: f(E|X) is %g at observation %d, so "
                "positivity fails in the sample and the functional is not "
                "identified there" % (p, i))

    out = {}
    if strategy in ("ye", "all"):
        tot = 0.0
        for i in range(n):
            if ev[i] == 0.0:
                tot += ey(i, 1.0) / (1.0 - fe1(i))
        out["ye"] = tot / n
    if strategy in ("em", "all"):
        tot = 0.0
        for i in range(n):
            if ev[i] == 1.0:
                d1 = fm(i, 1.0)
                d0 = fm(i, 0.0)
                if d1 <= 0.0:
                    raise ValueError(
                        "mediation_functional: f(M|E,X) is zero at "
                        "observation %d, so the inverse-odds weight is "
                        "undefined" % i)
                tot += yv[i] * (d0 / d1) / fe1(i)
        out["em"] = tot / n
    if strategy in ("ym", "all"):
        # plug-in: average over the observed X of the mediator-density-
        # weighted outcome regression, with M ranging over its support
        support = sorted(set(_cell_key(r) for r in Mm))
        tot = 0.0
        for i in range(n):
            s = 0.0
            for mkey in support:
                w = _density_at(fm, i, mkey, Mm, 0.0)
                if w > 0.0:
                    s += w * ey(i, 1.0, mkey)
            tot += s
        out["ym"] = tot / n
    if strategy != "all":
        return out[strategy]
    return out


def _saturated_models(ev, Mm, Xm):
    """Nonparametric cell estimates of f(E|X) and f(M|E,X)."""
    n = len(ev)
    xk = [_cell_key(r) for r in Xm]
    mk = [_cell_key(r) for r in Mm]
    nx, nx1 = {}, {}
    nem, ne = {}, {}
    for i in range(n):
        nx[xk[i]] = nx.get(xk[i], 0) + 1
        if ev[i] == 1.0:
            nx1[xk[i]] = nx1.get(xk[i], 0) + 1
        key = (ev[i], xk[i])
        ne[key] = ne.get(key, 0) + 1
        nem[(mk[i],) + key] = nem.get((mk[i],) + key, 0) + 1

    def fe1(i):
        return nx1.get(xk[i], 0) / float(nx[xk[i]])

    def fm(i, e, mkey=None):
        key = (mkey if mkey is not None else mk[i], e, xk[i])
        den = ne.get((e, xk[i]), 0)
        if den == 0:
            return 0.0
        return nem.get(key, 0) / float(den)

    return fe1, fm


def _saturated_outcome(yv, ev, Mm, Xm):
    """Nonparametric E(Y | E, M, X) as a cell mean."""
    n = len(yv)
    xk = [_cell_key(r) for r in Xm]
    mk = [_cell_key(r) for r in Mm]
    tot, cnt = {}, {}
    for i in range(n):
        key = (ev[i], mk[i], xk[i])
        tot[key] = tot.get(key, 0.0) + yv[i]
        cnt[key] = cnt.get(key, 0) + 1

    def ey(i, e, mkey=None):
        key = (e, mkey if mkey is not None else mk[i], xk[i])
        c = cnt.get(key, 0)
        return tot[key] / c if c else 0.0

    return ey


def _parametric_models(yv, ev, Mm, Xm, ridge):
    """Logistic f(E|X), Gaussian f(M|E,X), linear E(Y|E,M,X)."""
    n = len(yv)
    Ze = k.design(Xm, n)
    be = k.logit_irls(Ze, ev, 60, ridge)
    pe = [k.sigmoid(v) for v in k.matvec(Ze, be)]

    Zm = [[ev[i]] + list(Xm[i]) for i in range(n)]
    m1 = [row[0] for row in Mm]
    bm = k.lstsq(k.design(Zm, n), m1, ridge)
    Zm0 = [[0.0] + list(Xm[i]) for i in range(n)]
    Zm1 = [[1.0] + list(Xm[i]) for i in range(n)]
    mu0 = k.matvec(k.design(Zm0, n), bm)
    mu1 = k.matvec(k.design(Zm1, n), bm)
    muo = [mu1[i] if ev[i] == 1.0 else mu0[i] for i in range(n)]
    resid = [m1[i] - muo[i] for i in range(n)]
    s2 = sum(r * r for r in resid) / max(1, n - len(bm))
    if s2 <= 0.0:
        raise ValueError("mediation_functional: the mediator model has "
                         "zero residual variance")

    Zy = [[ev[i], m1[i]] + list(Xm[i]) for i in range(n)]
    by = k.lstsq(k.design(Zy, n), yv, ridge)

    def fe1(i):
        return pe[i]

    def fm(i, e, mkey=None):
        mval = m1[i] if mkey is None else float(mkey[0])
        mu = mu1[i] if e == 1.0 else mu0[i]
        r = mval - mu
        return math.exp(-0.5 * r * r / s2) / math.sqrt(2.0 * math.pi * s2)

    def ey(i, e, mkey=None):
        mval = m1[i] if mkey is None else float(mkey[0])
        row = [1.0, e, mval] + list(Xm[i])
        return sum(by[j] * row[j] for j in range(len(by)))

    return fe1, fm, ey


def _density_at(fm, i, mkey, Mm, e):
    return fm(i, e, mkey)


def weight_based_mediation(X, M, C, Y, strategy="em", saturated=True):
    r"""Natural direct and indirect effects by inverse-odds weighting.

    The argument order is the stub's, kept so callers do not break:
    `X` is the binary exposure (the paper's E), `M` the mediator, `C`
    the pre-exposure confounders (the paper's X), `Y` the outcome.

    Returns
    -------
    RichResult
        ``estimate`` is the natural indirect effect; ``nde``, ``nie``
        and ``theta`` are reported separately, along with every strategy
        when ``strategy="all"``.

    Examples
    --------
    A discrete saturated example, where all three strategies coincide::

        r = weight_based_mediation(E, M, C, Y, strategy="all")
        r["theta_em"], r["theta_ye"], r["theta_ym"]   # identical
    """
    ev = k.vec(X)
    yv = k.vec(Y)
    n = len(yv)
    theta = mediation_functional(yv, ev, M, C, strategy=strategy,
                                 saturated=saturated)
    thetas = theta if isinstance(theta, dict) else {strategy: theta}
    point = (thetas["em"] if "em" in thetas
             else list(thetas.values())[0])

    # E(Y_1) and E(Y_0) by the same nonparametric standardisation, so
    # the effects are internally consistent with theta.
    if saturated:
        fe1, _ = _saturated_models(ev, k.mat(M) if M is not None
                                   else [[0.0]] * n,
                                   k.mat(C) if C is not None
                                   else [[0.0]] * n)
    else:
        fe1, _, _ = _parametric_models(
            yv, ev, k.mat(M) if M is not None else [[0.0]] * n,
            k.mat(C) if C is not None else [[0.0]] * n, 1e-8)
    ey1 = sum(yv[i] * ev[i] / fe1(i) for i in range(n)) / n
    ey0 = sum(yv[i] * (1.0 - ev[i]) / (1.0 - fe1(i))
              for i in range(n)) / n

    out = {"estimate": ey1 - point,        # natural indirect effect
           "nie": ey1 - point,
           "nde": point - ey0,
           "theta": point,
           "ey1": ey1, "ey0": ey0,
           "total": ey1 - ey0,
           "n": n, "saturated": bool(saturated),
           "strategy": strategy,
           "method": "natural direct and indirect effects via the "
                     "mediation functional, Tchetgen Tchetgen & "
                     "Shpitser (2012) strategy %r" % (strategy,)}
    for kk, v in thetas.items():
        out["theta_" + kk] = v
    return RichResult(payload=out)


def cheatsheet():
    return ("wenge: mediation functional theta = E(Y_1,M_0) three ways "
            "(Tchetgen Tchetgen-Shpitser 2012). em = inverse-odds "
            "Y I(E=1)/f(E|X) * f(M|E=0,X)/f(M|E,X); ye = outcome model "
            "reweighted to the unexposed; ym = plug-in. Identical on a "
            "saturated model. NDE = theta - E(Y_0), NIE = E(Y_1) - theta.")


# compact alias per ledger/NAMING.md
weightbasedmediation = weight_based_mediation
