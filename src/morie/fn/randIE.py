# morie.fn -- function file (rootcoder007/morie)
r"""Randomized interventional direct and indirect effects.

Natural direct and indirect effects need a quantity no experiment can
produce: the outcome that *would* have occurred under one treatment
while the mediator took the value it *would* have taken under the
other. That is a cross-world statement -- two counterfactuals that
cannot both be realised -- and it is why natural effects require an
assumption no design can enforce.

**The randomized interventional analogue replaces the cross-world
quantity with a real intervention.** Rather than fixing the mediator at
the value a particular person would have had, draw it at random from
the mediator distribution that prevails under the other treatment arm:

.. math:: p(m \mid \sigma_M = s_{a^*}, c)
          = p(m \mid a^{*}, c).

This is Didelez, Dawid and Geneletti's *random (conditional)
intervention* regime: the strategy does not set :math:`M` to a value,
it sets :math:`M`'s conditional distribution. Because it is an
intervention on a distribution rather than on a cross-world pair, the
resulting estimand is well-defined for a policy someone could actually
run.

**The decomposition, and the price paid for it.** Writing

.. math:: \psi(a, a^{*}) =
          \sum_{c}\sum_{m} E[Y\mid a, m, c]\;
          p(m \mid a^{*}, c)\; p(c),

the total effect splits exactly:

.. math:: \underbrace{\psi(1,1)-\psi(0,0)}_{\text{total}}
          = \underbrace{\psi(1,0)-\psi(0,0)}_{\text{direct}}
          + \underbrace{\psi(1,1)-\psi(1,0)}_{\text{indirect}},

an algebraic identity that holds for any :math:`\psi` of this form,
whatever the data say -- so the anchor checks it as an identity and
does not treat agreement as evidence the estimator is right.

The price is that :math:`\psi(1,1)` is not the observed mean under
treatment unless the mediator distribution is degenerate: drawing
:math:`M` from :math:`p(m\mid a, c)` independently of the individual is
not the same as letting each individual keep their own :math:`M`. The
gap is exactly the covariance between the individual's mediator value
and their outcome surface, and it is reported rather than glossed.

**Identification is by the g-formula and nothing more exotic.** Under
no unmeasured confounding of the treatment-outcome, treatment-mediator
and mediator-outcome relations given :math:`C`, every term above is a
function of the observational distribution. That is the whole point of
the regime-indicator formulation: an estimand expressible in
observational conditionals is non-parametrically identified.

**Two estimation routes, because they fail differently.** The
``"gformula"`` route plugs in fitted conditional means and mediator
distributions; the ``"weighting"`` route reweights observed outcomes by
the ratio of mediator densities. The first is efficient when the
outcome model is right, the second survives an outcome model that is
wrong provided the mediator model is right. Both are exposed.

References
----------
Didelez, V., Dawid, A. P. & Geneletti, S. (2006) "Direct and Indirect
Effects of Sequential Treatments", in *Proceedings of the 22nd
Conference on Uncertainty in Artificial Intelligence (UAI 2006)*,
138-146, arXiv:1206.6840. Sec. 2.1 (regime indicators; the
observational, atomic, conditional and random-conditional regimes),
Sec. 3 (direct and indirect effects defined by intervention) and
Sec. 4 (identification from observational data via the g-formula).

Dawid, A. P. & Didelez, V. (2010) "Identifying the consequences of
dynamic treatment strategies: A decision-theoretic overview",
*Statistics Surveys* 4, 184-231, doi:10.1214/10-SS081. The
regime-indicator framework the above builds on.

Robins, J. M. (1986) "A new approach to causal inference in mortality
studies with a sustained exposure period", *Mathematical Modelling*
7(9-12), 1393-1512, doi:10.1016/0270-0255(86)90088-6. The g-formula
used for identification.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["mediator_distribution", "interventional_mean",
           "randomized_interventional_effect", "decompose"]

_EPS = 1e-12
_ROUTES = ("gformula", "weighting")


def _labels(v, name):
    out = [str(x) for x in v]
    if not out:
        raise ValueError("randIE: %s is empty" % name)
    return out


def mediator_distribution(A, M, C=None, laplace=0.0):
    r"""Empirical :math:`p(m \mid a, c)`, the regime being intervened
    upon.

    ``laplace`` adds a constant to each cell count. It is off by
    default: a smoothed distribution is a different estimand, and
    turning it on silently would hide an empty stratum rather than
    report one.
    """
    a = _labels(A, "A")
    m = _labels(M, "M")
    n = len(a)
    if len(m) != n:
        raise ValueError("randIE: %d treatments but %d mediator "
                         "values" % (n, len(m)))
    c = ["*"] * n if C is None else _labels(C, "C")
    if len(c) != n:
        raise ValueError("randIE: %d strata for %d units"
                         % (len(c), n))
    levels = sorted(set(m))
    cells = {}
    for i in range(n):
        cells.setdefault((a[i], c[i]), []).append(m[i])
    out = {}
    for key, vals in cells.items():
        tot = len(vals) + float(laplace) * len(levels)
        out[key] = {lv: (vals.count(lv) + float(laplace)) / tot
                    for lv in levels}
    return {"p": out, "levels": levels,
            "strata": sorted({cc for _, cc in cells}),
            "arms": sorted({aa for aa, _ in cells}),
            "n": n}


def interventional_mean(Y, A, M, C=None, a="1", a_star="0",
                        route="gformula", laplace=0.0):
    r"""The functional :math:`\psi(a, a^{*})`.

    Outcomes are taken under treatment ``a``; the mediator is drawn
    from the distribution it has under ``a_star``. Setting
    ``a == a_star`` gives the mean under a random-mediator version of
    that arm, which is **not** in general the observed arm mean -- see
    ``own_mediator_mean`` in the result.
    """
    if route not in _ROUTES:
        raise ValueError("randIE: route must be gformula or "
                         "weighting, got %r" % (route,))
    y = [float(v) for v in k.vec(Y)]
    av = _labels(A, "A")
    mv = _labels(M, "M")
    n = len(y)
    if not (len(av) == len(mv) == n):
        raise ValueError("randIE: Y, A and M must agree in length "
                         "(%d, %d, %d)" % (n, len(av), len(mv)))
    cv = ["*"] * n if C is None else _labels(C, "C")
    if len(cv) != n:
        raise ValueError("randIE: %d strata for %d units"
                         % (len(cv), n))
    a, a_star = str(a), str(a_star)
    if a not in set(av):
        raise ValueError("randIE: treatment arm %r not observed; arms "
                         "are %s" % (a, sorted(set(av))))
    if a_star not in set(av):
        raise ValueError("randIE: treatment arm %r not observed; arms "
                         "are %s" % (a_star, sorted(set(av))))
    md = mediator_distribution(av, mv, cv, laplace=laplace)
    strata = md["strata"]
    levels = md["levels"]
    pc = {s: sum(1 for i in range(n) if cv[i] == s) / float(n)
          for s in strata}

    # E[Y | a, m, c] from the cell means
    ybar, cnt = {}, {}
    for i in range(n):
        key = (av[i], mv[i], cv[i])
        ybar[key] = ybar.get(key, 0.0) + y[i]
        cnt[key] = cnt.get(key, 0) + 1
    for key in ybar:
        ybar[key] /= cnt[key]

    total, missing = 0.0, []
    for s in strata:
        pm = md["p"].get((a_star, s))
        if pm is None:
            missing.append(("mediator distribution", a_star, s))
            continue
        for lv in levels:
            w = pm[lv]
            if w <= 0.0:
                continue
            key = (a, lv, s)
            if key not in ybar:
                missing.append(("outcome mean", a, lv, s))
                continue
            total += pc[s] * w * ybar[key]
    if missing:
        raise ValueError("randIE: %d cell(s) needed by the g-formula "
                         "are empty, e.g. %r -- psi(%s, %s) is not "
                         "identified from this sample"
                         % (len(missing), missing[0], a, a_star))

    if route == "weighting":
        # reweight observed outcomes in arm a by p(m | a*, c)/p(m | a, c)
        num, den = 0.0, 0.0
        for i in range(n):
            if av[i] != a:
                continue
            p_star = md["p"].get((a_star, cv[i]), {}).get(mv[i], 0.0)
            p_obs = md["p"].get((a, cv[i]), {}).get(mv[i], 0.0)
            if p_obs <= _EPS:
                continue
            w = p_star / p_obs
            num += w * y[i]
            den += w
        if den <= _EPS:
            raise ValueError("randIE: the mediator-density ratio put "
                             "no weight on arm %r" % (a,))
        total = num / den

    own = ([y[i] for i in range(n) if av[i] == a])
    return {"estimate": total, "a": a, "a_star": a_star,
            "route": route,
            "own_mediator_mean": sum(own) / len(own) if own else
            float("nan"),
            "n_arm": len(own), "n": n,
            "note": "psi(a, a) is not the observed arm mean unless the "
                    "mediator is degenerate: drawing M from p(m|a,c) "
                    "breaks the individual-level M-Y dependence"}


def randomized_interventional_effect(Y, A, M, C=None, treated="1",
                                     control="0", route="gformula",
                                     laplace=0.0):
    r"""Total, direct and indirect effects under the random regime."""
    def psi(a, a_star):
        return interventional_mean(Y, A, M, C, a=a, a_star=a_star,
                                   route=route, laplace=laplace
                                   )["estimate"]
    t, c = str(treated), str(control)
    p11, p10, p00 = psi(t, t), psi(t, c), psi(c, c)
    # psi(control, treated) is only needed for the control-arm direct
    # effect, which is a diagnostic. It asks for E[Y | a=control, m, c]
    # at mediator values the control arm may never take, and when that
    # cell is empty the quantity is genuinely unidentified -- so it is
    # reported as None rather than allowed to sink the decomposition,
    # which needs only the three above.
    try:
        p01 = psi(c, t)
    except ValueError:
        p01 = None
    return RichResult(payload={
        "estimate": p11 - p00,
        "total": p11 - p00,
        "direct": p10 - p00,
        "indirect": p11 - p10,
        "direct_control_arm": (p00 - p01) if p01 is not None
                              else None,
        "psi": {"11": p11, "10": p10, "01": p01, "00": p00},
        "route": route, "treated": t, "control": c,
        "identity": "total = direct + indirect holds exactly by "
                    "construction; it is not evidence the estimator "
                    "is correct",
        "method": "randomized interventional direct/indirect effects, "
                  "Didelez, Dawid & Geneletti (2006) Secs. 3-4",
    })


def decompose(result):
    """Total / direct / indirect with the residual, which must be 0."""
    tot = result["total"]
    d, i = result["direct"], result["indirect"]
    return {"total": tot, "direct": d, "indirect": i,
            "residual": tot - (d + i),
            "proportion_mediated": (i / tot) if abs(tot) > _EPS
            else float("nan")}


def cheatsheet():
    return ("randIE: randomized interventional effects. Natural "
            "effects need a CROSS-WORLD quantity no experiment can "
            "produce; here the mediator is instead DRAWN from "
            "p(m|a*,c) -- an intervention on a distribution, so the "
            "estimand is a runnable policy. psi(a,a*) = sum_c sum_m "
            "E[Y|a,m,c] p(m|a*,c) p(c); total = direct + indirect is "
            "an algebraic IDENTITY, not a check on the fit. psi(a,a) "
            "is NOT the observed arm mean.")


# compact alias per ledger/NAMING.md
randomizedinterventionaleffect = randomized_interventional_effect
