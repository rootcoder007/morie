# morie.fn -- function file (rootcoder007/morie)
r"""Evaluating a treatment prioritization rule: TOC, RATE, Qini.

A CATE model is usually judged by how well it predicts the effect. But
the thing a decision maker does with it is *rank* people -- treat the
top decile, treat everyone with a positive score -- and a model can
rank well while predicting badly, or predict well and rank badly. This
module scores the ranking, which is the quantity the decision actually
depends on.

**A prioritization rule is just a score.** :math:`S : \mathcal X \to
\mathbb R`, units treated in decreasing order of :math:`S(X_i)`
(Definition 1). Nothing requires :math:`S` to be a CATE estimate: it
may be a risk score, a hand-written heuristic, or a learned rule. That
is the point -- it puts risk-based and CATE-based rules on one
yardstick.

**The targeting operator characteristic** is the gain from treating
the top :math:`u` fraction rather than everyone (Definition 2):

.. math:: \mathrm{TOC}(u; S) =
          E\!\left[Y_i(1) - Y_i(0) \mid F_S(S(X_i)) \ge 1-u\right]
          - E\!\left[Y_i(1) - Y_i(0)\right].

At :math:`u = 1` the first term *is* the ATE, so :math:`\mathrm{TOC}(1)
= 0` identically -- a fixed point the implementation has to reproduce,
not approximate.

**The RATE is a weighted average of it** (Definition 3),
:math:`\theta_\alpha(S) = \int_0^1 \alpha(u)\,\mathrm{TOC}(u; S)\,du`,
and the two standard choices are weights, not different metrics:

* :math:`\alpha(u) = u` gives the **Qini coefficient** of Radcliffe;
* :math:`\alpha(u) = 1` gives the **AUTOC** of Zhao et al.

**Which weight to use is a power question with a known answer.** If the
treatment effect is non-zero for a large share of the population, the
linear Qini weight has more power; if the benefit is concentrated in a
small subgroup, the logarithmic AUTOC weighting does (Sec. 4, Fig. 2).
Both are exposed and neither is hidden behind a default.

**The null that matters is exactly right.** If :math:`S(X_i)` is
independent of :math:`Y_i(1)-Y_i(0)` then the TOC and every RATE are
identically zero (Remark 1). So testing :math:`\theta = 0` is a test of
whether the rule has found real heterogeneity -- and it is *not* a test
of whether the ATE is non-zero, which is a different question that a
useless rule can still pass.

**Estimation uses doubly-robust scores, so it works off an RCT.** With

.. math:: \hat\Gamma_i = \hat\mu_1(X_i) - \hat\mu_0(X_i)
          + \frac{W_i(Y_i - \hat\mu_1(X_i))}{\hat e(X_i)}
          - \frac{(1-W_i)(Y_i - \hat\mu_0(X_i))}{1-\hat e(X_i)}

the AIPW score is unbiased for :math:`\tau(X_i)` when either the
outcome models or the propensity is right, and the TOC is then a
running mean of :math:`\hat\Gamma` down the ranking minus its grand
mean. Inference is by the half-sample bootstrap, which Corollary 5
justifies from asymptotic linearity alone.

References
----------
Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E. & Wager, S. (2025)
"Evaluating Treatment Prioritization Rules via Rank-Weighted Average
Treatment Effects", *Journal of the American Statistical Association*
120(549), 38-51, doi:10.1080/01621459.2024.2393466. Definition 1
(prioritization rule), Definition 2 (TOC), Definition 3 (RATE),
Remark 1 (the exact null), Sec. 2.2-2.3 (the AIPW-score estimator),
Theorem 3 and Corollary 5 (asymptotic linearity and the half-sample
bootstrap), Sec. 4 and Fig. 2 (Qini vs AUTOC power).

Sverdrup, E., Wu, H., Athey, S. & Wager, S. (2025) "Qini Curves for
Multi-Armed Treatment Rules", *Journal of Computational and Graphical
Statistics* 34(3), 948-960, doi:10.1080/10618600.2024.2418820. The
Qini curve under a cost constraint and its multi-armed generalisation.

Athey, S., Tibshirani, J. & Wager, S. (2019) "Generalized random
forests", *The Annals of Statistics* 47(2), 1148-1178,
doi:10.1214/18-AOS1709. The forest whose CATE estimates are the usual
priority score here.

Notes
-----
The ledger recorded this module as "Athey-Wager (2024), sliced GRF for
cross-sectional CATE". No paper of that title was located; the entry
came from the generated stub. The implementation follows
Yadlowsky et al. (2025) and Sverdrup et al. (2025).
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["aipw_scores", "toc_curve", "rate", "qini_coefficient",
           "autoc", "qini_curve", "rate_test"]

_EPS = 1e-12
_WEIGHTS = ("qini", "autoc", "uniform")


def _check(scores, priority):
    g = [float(v) for v in k.vec(scores)]
    s = [float(v) for v in k.vec(priority)]
    if len(g) != len(s):
        raise ValueError("slvgrf: %d scores but %d priority values"
                         % (len(g), len(s)))
    if len(g) < 2:
        raise ValueError("slvgrf: need at least 2 units, got %d"
                         % len(g))
    return g, s


def aipw_scores(Y, W, mu1, mu0, e):
    r"""The doubly-robust score :math:`\hat\Gamma_i` of Sec. 2.3.

    Unbiased for :math:`\tau(X_i)` if either the outcome regressions
    ``mu1``/``mu0`` or the propensity ``e`` is correct. In a randomised
    trial ``e`` is known, so the score is unbiased whatever the outcome
    models do -- which is why this is the default route.
    """
    y = [float(v) for v in k.vec(Y)]
    w = [float(v) for v in k.vec(W)]
    m1 = [float(v) for v in k.vec(mu1)]
    m0 = [float(v) for v in k.vec(mu0)]
    n = len(y)
    ev = ([float(e)] * n if isinstance(e, (int, float))
          else [float(v) for v in k.vec(e)])
    for nm, v in (("W", w), ("mu1", m1), ("mu0", m0), ("e", ev)):
        if len(v) != n:
            raise ValueError("slvgrf: %s has %d entries for %d units"
                             % (nm, len(v), n))
    for v in w:
        if v not in (0.0, 1.0):
            raise ValueError("slvgrf: W must be 0/1, got %r" % (v,))
    for v in ev:
        if not 0.0 < v < 1.0:
            raise ValueError("slvgrf: the propensity must lie strictly "
                             "in (0, 1); got %r -- overlap fails"
                             % (v,))
    return [m1[i] - m0[i]
            + w[i] * (y[i] - m1[i]) / ev[i]
            - (1.0 - w[i]) * (y[i] - m0[i]) / (1.0 - ev[i])
            for i in range(n)]


def toc_curve(scores, priority):
    r"""TOC at :math:`u = j/n`, :math:`j = 1..n` (Definition 2).

    The value at :math:`u = j/n` is the mean score over the top
    :math:`j` units by priority, minus the grand mean. The last point
    is therefore exactly zero.
    """
    g, s = _check(scores, priority)
    n = len(g)
    order = sorted(range(n), key=lambda i: (-s[i], i))
    ate = sum(g) / n
    run, toc, us = 0.0, [], []
    for j, i in enumerate(order, start=1):
        run += g[i]
        toc.append(run / j - ate)
        us.append(j / float(n))
    return {"u": us, "toc": toc, "ate": ate, "order": order, "n": n}


def rate(scores, priority, weight="autoc"):
    r"""RATE :math:`\int_0^1 \alpha(u)\,\mathrm{TOC}(u)\,du`.

    ``weight="qini"`` uses :math:`\alpha(u) = u` (the Qini
    coefficient), ``"autoc"`` uses :math:`\alpha(u) = 1` (the AUTOC of
    Zhao et al.), ``"uniform"`` is a synonym for ``"autoc"`` kept
    because the literature names the same weight both ways.
    """
    if weight not in _WEIGHTS:
        raise ValueError("slvgrf: weight must be one of %s, got %r"
                         % (", ".join(_WEIGHTS), weight))
    c = toc_curve(scores, priority)
    n = c["n"]
    if weight == "qini":
        val = sum(c["u"][j] * c["toc"][j] for j in range(n)) / n
    else:
        val = sum(c["toc"]) / n
    return {"estimate": val, "weight": weight, "curve": c, "n": n}


def autoc(scores, priority):
    """Area under the TOC -- the RATE with a flat weight."""
    return rate(scores, priority, weight="autoc")["estimate"]


def qini_coefficient(scores, priority):
    r"""The Qini coefficient -- the RATE with :math:`\alpha(u) = u`."""
    return rate(scores, priority, weight="qini")["estimate"]


def qini_curve(scores, priority, cost=None):
    r"""Cumulative gain from treating the top fraction.

    :math:`Q(u) = u \cdot E[\tau \mid \text{top } u]`, the quantity
    plotted in the marketing literature. With a per-unit ``cost`` the
    horizontal axis becomes spend rather than headcount, which is the
    constrained version Sverdrup et al. study; the curve is then read
    against the budget actually available.
    """
    g, s = _check(scores, priority)
    n = len(g)
    order = sorted(range(n), key=lambda i: (-s[i], i))
    if cost is None:
        cv = [1.0] * n
    else:
        cv = ([float(cost)] * n if isinstance(cost, (int, float))
              else [float(v) for v in k.vec(cost)])
        if len(cv) != n:
            raise ValueError("slvgrf: %d costs for %d units"
                             % (len(cv), n))
        if any(v <= 0.0 for v in cv):
            raise ValueError("slvgrf: costs must be positive")
    total = sum(cv)
    run, spent, xs, ys = 0.0, 0.0, [], []
    for i in order:
        run += g[i]
        spent += cv[i]
        xs.append(spent / total)
        ys.append(run / n)
    return {"spend": xs, "gain": ys, "ate": sum(g) / n, "n": n,
            "constrained": cost is not None}


def rate_test(scores, priority, weight="autoc", reps=500, seed=0):
    r"""Half-sample bootstrap test of :math:`\theta_\alpha(S) = 0`.

    Corollary 5: the RATE is asymptotically linear, so an estimate on a
    random half-sample (drawn without replacement) has the same
    limiting distribution around :math:`\hat\theta` that
    :math:`\hat\theta` has around :math:`\theta`. The half-sample
    spread is therefore rescaled by :math:`1/\sqrt 2` to give the
    standard error of the full-sample estimate.

    Under the null of Remark 1 -- the priority score independent of the
    treatment effect -- the true RATE is zero, so this is a test of
    whether the rule found heterogeneity, **not** of whether the ATE is
    non-zero.
    """
    g, s = _check(scores, priority)
    n = len(g)
    if n < 8:
        raise ValueError("slvgrf: the half-sample bootstrap needs at "
                         "least 8 units, got %d" % n)
    theta = rate(g, s, weight=weight)["estimate"]
    rng = np.random.default_rng(seed)
    half = n // 2
    draws = []
    for _ in range(int(reps)):
        idx = sorted(range(n), key=lambda _i: float(rng.uniform()))[:half]
        draws.append(rate([g[i] for i in idx], [s[i] for i in idx],
                          weight=weight)["estimate"])
    m = sum(draws) / len(draws)
    v = sum((d - m) ** 2 for d in draws) / (len(draws) - 1)
    se = math.sqrt(max(v, 0.0) / 2.0)
    z = theta / se if se > _EPS else 0.0
    p = 2.0 * (1.0 - k.pnorm(abs(z)))
    return RichResult(payload={
        "estimate": theta, "se": se, "z": z, "p_value": p,
        "weight": weight, "reps": int(reps), "n": n,
        "null": "the priority score is independent of the treatment "
                "effect (Remark 1), NOT that the ATE is zero",
        "method": "RATE with half-sample bootstrap, Yadlowsky et al. "
                  "(2025) Corollary 5",
    })


def cheatsheet():
    return ("slvgrf: score a PRIORITIZATION RULE, not a CATE fit. "
            "TOC(u) = mean effect in the top u minus the ATE, so "
            "TOC(1) = 0 exactly. RATE = int alpha(u) TOC(u) du; "
            "alpha(u)=u is Qini, alpha(u)=1 is AUTOC. If the score is "
            "independent of the effect, every RATE is exactly 0 -- so "
            "this tests HETEROGENEITY, not the ATE. Qini has more "
            "power when many units benefit, AUTOC when few do. "
            "Estimate off AIPW scores; test by half-sample bootstrap.")


# compact alias per ledger/NAMING.md
slicedgrf = rate
