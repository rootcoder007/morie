# morie.fn -- function file (rootcoder007/morie)
r"""Online targeted learning for a single time series.

At each time we observe, in order, a covariate vector, a treatment and
an outcome. The conditional law of that triple given the past depends
on the past only through a **fixed-dimensional summary measure**, and
the mechanism producing it is constant in time. That is the whole
statistical model, and its generality is the point: an empty summary
measure gives ordinary i.i.d. targeted learning; a parametric
conditional density gives a classical time series model; a
data-dependent randomisation gives a group sequential adaptive design.

**Effects are defined by stochastic interventions on future treatment
nodes.** With a single time series there is no population to average
over, so a static "set :math:`A_t = 1` for everyone" has no referent.
The causal quantity is instead the mean of a future outcome under a
stochastic intervention on a *subset* of the treatment nodes, and the
chapter establishes that these are identifiable from the observed data
distribution.

**Where the sample size comes from.** Not from independent units --
there is one series. It comes from **time**: the fixed-dimensional
summary and the time-invariant mechanism mean each new time point is
another draw from the same conditional law. So asymptotics are in
:math:`t`, and the influence curve is a **martingale** difference
sequence rather than an i.i.d. sum. Its variance is estimated by the
sum of conditional variances, and the martingale central limit theorem
supplies the normal limit.

**Which makes one check essential.** If the summary measure is too
small, the "past" left out is still influencing the present, the
martingale property fails, and the reported interval is wrong for
reasons no amount of data fixes. ``martingale_check`` regresses the
influence terms on the past and reports the dependence that should not
be there.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 19 (van der
Laan, Chambaz & Lendle): a time series in which one observes in
chronological order a covariate vector, a treatment and an outcome;
the conditional distribution given the past depending on the past
through a fixed-dimensional summary measure, described by a
time-invariant mechanism in a model space that may be unspecified; a
compatible causal model with a family of causal effects defined by
STOCHASTIC INTERVENTIONS on a subset of the treatment nodes on a
future outcome, and their identifiability from the observed data
distribution; and the observation that empty summary measures recover
i.i.d. targeted learning, parametric conditional densities recover
classical time series models, and group sequential adaptive designs
are included.

Chambaz, A., Zheng, W. & van der Laan, M. J. (2017) "Targeted
sequential design for targeted learning inference of the optimal
treatment rule and its mean reward", *Annals of Statistics* 45(6),
2537-2564, doi:10.1214/16-AOS1534.

van der Laan, M. J., Rose, S. & Lendle, S. (2018) "Online Targeted
Learning for Time Series", in *Targeted Learning in Data Science*,
Springer, doi:10.1007/978-3-319-65304-4_19.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["lag_summary", "stochastic_intervention",
           "martingale_variance", "martingale_check",
           "online_tmle_series"]

_EPS = 1e-12


def lag_summary(series, t, lags=2):
    r"""The fixed-dimensional summary :math:`Z(t)` of the past."""
    v = [float(q) for q in k.vec(series)]
    L = int(lags)
    if L < 0:
        raise ValueError("tlonts: lags must be non-negative")
    if t <= 0:
        return [0.0] * L
    past = v[max(0, t - L):t]
    return [0.0] * (L - len(past)) + past


def stochastic_intervention(A, nodes, shift=None, prob=None):
    r"""Intervene on a SUBSET of the treatment nodes.

    ``shift`` moves the treatment; ``prob`` replaces its distribution
    with a Bernoulli. Nodes outside the subset keep their observed
    values, which is what makes the intervention a statement about a
    future segment rather than the whole series.
    """
    a = [float(v) for v in k.vec(A)]
    idx = set(int(v) for v in nodes)
    if any(v < 0 or v >= len(a) for v in idx):
        raise ValueError("tlonts: an intervention node is outside the "
                         "series")
    if (shift is None) == (prob is None):
        raise ValueError("tlonts: give exactly one of shift or prob")
    out = list(a)
    for i in idx:
        out[i] = a[i] + float(shift) if shift is not None \
            else float(prob)
    return {"intervened": out, "nodes": sorted(idx),
            "n_intervened": len(idx),
            "kind": "shift" if shift is not None else "bernoulli"}


def martingale_variance(D):
    r"""Variance from the sum of squares of the martingale
    differences.

    Not the i.i.d. variance: the terms are dependent, but a martingale
    difference sequence has uncorrelated terms, so the sum of squares
    is still the right estimator.
    """
    v = [float(q) for q in k.vec(D)]
    T = len(v)
    if T < 2:
        raise ValueError("tlonts: at least 2 time points are needed")
    s2 = sum(q * q for q in v) / T
    return {"variance": s2, "se": math.sqrt(s2 / T), "T": T,
            "note": "asymptotics are in TIME, not in independent "
                    "units"}


def martingale_check(D, past, tol=0.2):
    r"""Is the influence term really unpredictable from the past?

    Correlation with the past means the summary measure is too small
    -- the martingale property fails, and no amount of data repairs
    the interval.
    """
    d = [float(q) for q in k.vec(D)]
    p = [float(q) for q in k.vec(past)]
    if len(d) != len(p):
        raise ValueError("tlonts: %d influence terms but %d past "
                         "values" % (len(d), len(p)))
    n = len(d)
    md, mp = sum(d) / n, sum(p) / n
    num = sum((d[i] - md) * (p[i] - mp) for i in range(n))
    den = math.sqrt(sum((d[i] - md) ** 2 for i in range(n))
                    * sum((p[i] - mp) ** 2 for i in range(n)))
    r = num / den if den > _EPS else 0.0
    return {"correlation": r, "is_martingale": abs(r) < float(tol),
            "note": "a non-zero correlation says the summary measure "
                    "omits something the present still depends on"}


def online_tmle_series(Y, A, Z, Q_fn, g_fn, target_prob, burn_in=10):
    r"""Sequentially updated TMLE for a stochastic intervention.

    The clever covariate is the ratio of the intervened to the
    observed treatment probability, and the estimate is updated as
    each time point arrives.
    """
    y = [float(v) for v in k.vec(Y)]
    a = [float(v) for v in k.vec(A)]
    z = [[float(v) for v in r] for r in k.mat(Z)]
    T = len(y)
    if not (len(a) == len(z) == T):
        raise ValueError("tlonts: the series differ in length")
    b = int(burn_in)
    if b < 1 or b >= T:
        raise ValueError("tlonts: burn_in must lie in 1..%d" % (T - 1))
    est, D = [], []
    running = 0.0
    for t in range(b, T):
        g = float(g_fn(z[t]))
        if g <= 0.0 or g >= 1.0:
            raise ValueError("tlonts: the treatment probability left "
                             "(0,1) at time %d" % t)
        gs = float(target_prob)
        h = (gs / g) if a[t] == 1.0 else ((1.0 - gs) / (1.0 - g))
        q1 = float(Q_fn(1.0, z[t]))
        q0 = float(Q_fn(0.0, z[t]))
        qa = q1 if a[t] == 1.0 else q0
        psi_t = gs * q1 + (1.0 - gs) * q0
        running += psi_t
        cur = running / (t - b + 1)
        est.append(cur)
        D.append(h * (y[t] - qa) + psi_t - cur)
    mv = martingale_variance(D)
    return RichResult(payload={
        "estimate": est[-1], "psi": est[-1], "path": est,
        "se": mv["se"], "ci": (est[-1] - 1.96 * mv["se"],
                               est[-1] + 1.96 * mv["se"]),
        "T_scored": mv["T"],
        "method": "online TMLE for a time series under a stochastic "
                  "intervention; van der Laan & Rose (2018) Chap. 19",
        "note": "one series, no independent units -- the sample size "
                "is TIME, and the influence terms form a martingale "
                "difference sequence",
    })


def cheatsheet():
    return ("tlonts: ONE time series -- covariate, treatment, outcome "
            "at each step -- with the conditional law depending on the "
            "past only through a FIXED-DIMENSIONAL summary and a "
            "time-invariant mechanism. Effects are defined by "
            "STOCHASTIC interventions on a SUBSET of future treatment "
            "nodes, since with a single series there is no population "
            "to set treatment for. Sample size comes from TIME: the "
            "influence terms are a MARTINGALE difference sequence, so "
            "variance is the sum of squares and the CLT is the "
            "martingale one. If the summary is too small the "
            "martingale property fails and the interval is simply "
            "wrong.")


# compact alias per ledger/NAMING.md
onlinetimeseriestmle = online_tmle_series
