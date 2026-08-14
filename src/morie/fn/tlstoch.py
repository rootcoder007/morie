# morie.fn -- function file (rootcoder007/morie)
r"""Stochastic treatment regimes.

Static regimes assign one treatment level to everyone; dynamic regimes
assign it as a function of measured history. Both are **deterministic**
-- fully fixed by pre-treatment variables -- and both are the wrong
frame for a great deal of applied work.

**Two reasons the chapter gives, and they are different.** First,
realistic interventions often cannot put treatment into a deterministic
state: setting an individual's exercise regime by a fixed rule is not
something anyone can do. A mass-media campaign is deterministic at the
community level and *stochastic at the individual level*, because each
person adopts or not for reasons outside the intervention. Second,
even where a deterministic regime is conceivable, its effect may be
**unidentifiable** because the regime is not supported in the observed
data -- nobody in the data behaves that way.

**A stochastic regime shifts the treatment distribution instead of
setting it.** For a continuous exposure the natural version is a
*shift*: :math:`d(a, w) = a + \delta`, so everyone's exposure moves by
:math:`\delta` from wherever it was. The estimand is the mean outcome
under the shifted distribution,

.. math:: \Psi(P) = E\Big[\int \bar Q(a, W)\, g_\delta(a \mid W)\,
          da\Big],

and for the shift intervention this equals
:math:`E[\bar Q(A + \delta, W)]` -- an average of the outcome
regression over the *observed* exposure distribution, translated.

**Positivity becomes a support condition, and a milder one.** A
deterministic regime needs positive probability of the assigned value
for every covariate pattern; a shift needs only that the shifted value
stays inside the support of the conditional exposure distribution.
``positivity_shift`` reports the fraction that leaves it, because the
whole appeal of a stochastic regime is lost if the shift walks off the
edge of the data.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 14 (Diaz &
van der Laan): static and dynamic regimes are both deterministic
because they are completely determined by pre-treatment variables;
deterministic regimes are the wrong framework for phenomena not
subject to direct intervention -- the exercise-regime and mass-media
examples, where an intervention is deterministic at the community
level and stochastic at the individual level; causal effects for
deterministic regimes may be unidentifiable because the regime is not
supported in the observed data; the data, notation and parameter of
interest with its identification and positivity assumption; the
optimality theory for stochastic regimes; the TMLE and its asymptotic
distribution; and super learning for a conditional density as the
initial estimator.

Diaz, I. & van der Laan, M. J. (2012) "Population Intervention Causal
Effects Based on Stochastic Interventions", *Biometrics* 68(2),
541-549, doi:10.1111/j.1541-0420.2011.01685.x.

Haneuse, S. & Rotnitzky, A. (2013) "Estimation of the effect of
interventions that modify the received treatment", *Statistics in
Medicine* 32(30), 5260-5277, doi:10.1002/sim.5907. Modified-treatment
policies.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["shift_regime", "positivity_shift", "stochastic_estimand",
           "shift_tmle", "density_ratio"]

_EPS = 1e-12


def shift_regime(A, delta, lower=None, upper=None):
    r"""The modified exposure :math:`d(a) = a + \delta`.

    Truncated at the support bounds when given, since a shift beyond
    them is not an intervention the data can speak to.
    """
    a = [float(v) for v in k.vec(A)]
    d = float(delta)
    out, clipped = [], 0
    for v in a:
        s = v + d
        if lower is not None and s < float(lower):
            s = float(lower)
            clipped += 1
        if upper is not None and s > float(upper):
            s = float(upper)
            clipped += 1
        out.append(s)
    return {"shifted": out, "delta": d, "n_clipped": clipped,
            "fraction_clipped": clipped / float(len(a))}


def positivity_shift(A, delta, W=None, bins=5):
    r"""How much of the shifted exposure leaves the observed support.

    For a stochastic regime this replaces the usual positivity
    condition -- and it is milder, because the shift only has to stay
    inside the conditional support rather than pile all mass on one
    value.
    """
    a = [float(v) for v in k.vec(A)]
    d = float(delta)
    if W is None:
        lo, hi = min(a), max(a)
        out = sum(1 for v in a if v + d < lo or v + d > hi)
        return {"fraction_outside": out / float(len(a)),
                "support": (lo, hi), "delta": d,
                "satisfied": out == 0}
    w = [float(v) for v in k.vec(W)]
    lo_w, hi_w = min(w), max(w)
    width = (hi_w - lo_w) / int(bins) or 1.0
    out = 0
    for i in range(len(a)):
        b = min(int((w[i] - lo_w) / width), int(bins) - 1)
        same = [a[j] for j in range(len(a))
                if min(int((w[j] - lo_w) / width),
                       int(bins) - 1) == b]
        if a[i] + d < min(same) or a[i] + d > max(same):
            out += 1
    return {"fraction_outside": out / float(len(a)), "delta": d,
            "bins": int(bins), "satisfied": out == 0,
            "note": "milder than deterministic positivity: the shift "
                    "only has to stay inside the CONDITIONAL support"}


def stochastic_estimand(Q_fn, A, W, delta, lower=None, upper=None):
    r"""The g-computation estimand under the shift.

    :math:`\Psi = E[\bar Q(A+\delta, W)]` -- the outcome regression
    averaged over the OBSERVED exposure distribution, translated.
    """
    a = [float(v) for v in k.vec(A)]
    rows = [[float(v) for v in r] for r in k.mat(W)]
    if len(rows) != len(a):
        raise ValueError("tlstoch: %d exposures but %d covariate rows"
                         % (len(a), len(rows)))
    sh = shift_regime(a, delta, lower, upper)["shifted"]
    vals = [float(Q_fn(sh[i], rows[i])) for i in range(len(a))]
    obs = [float(Q_fn(a[i], rows[i])) for i in range(len(a))]
    return {"psi": sum(vals) / len(vals),
            "observed_mean": sum(obs) / len(obs),
            "contrast": sum(vals) / len(vals) - sum(obs) / len(obs),
            "delta": float(delta)}


def density_ratio(A, W, delta, g_fn, lower=None, upper=None):
    r"""The clever covariate for a shift:
    :math:`g(a-\delta\mid w)/g(a\mid w)`.

    A ratio of densities rather than an inverse probability, which is
    why the estimator degrades gracefully where a deterministic
    regime's clever covariate would explode.
    """
    a = [float(v) for v in k.vec(A)]
    rows = [[float(v) for v in r] for r in k.mat(W)]
    d = float(delta)
    out = []
    for i in range(len(a)):
        num = float(g_fn(a[i] - d, rows[i]))
        den = float(g_fn(a[i], rows[i]))
        if den <= _EPS:
            raise ValueError("tlstoch: the observed exposure has zero "
                             "density at observation %d -- the "
                             "conditional density estimate is "
                             "degenerate" % i)
        out.append(num / den)
    return {"H": out, "max": max(out), "mean": sum(out) / len(out)}


def shift_tmle(Y, A, W, Q_fn, g_fn, delta, lower=None, upper=None,
               iters=60):
    r"""TMLE of the mean outcome under a shift intervention."""
    y = [float(v) for v in k.vec(Y)]
    a = [float(v) for v in k.vec(A)]
    rows = [[float(v) for v in r] for r in k.mat(W)]
    n = len(y)
    H = density_ratio(a, rows, delta, g_fn, lower, upper)["H"]
    q = [float(Q_fn(a[i], rows[i])) for i in range(n)]
    e = 0.0
    for _ in range(int(iters)):
        pred = [q[i] + e * H[i] for i in range(n)]
        gr = sum(H[i] * (y[i] - pred[i]) for i in range(n))
        he = sum(H[i] * H[i] for i in range(n))
        if he < 1e-12:
            break
        step = gr / he
        e += step
        if abs(step) < 1e-12:
            break
    sh = shift_regime(a, delta, lower, upper)["shifted"]
    qs = [float(Q_fn(sh[i], rows[i])) + e for i in range(n)]
    psi = sum(qs) / n
    d = [H[i] * (y[i] - (q[i] + e * H[i])) + qs[i] - psi
         for i in range(n)]
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / n ** 2)
    return RichResult(payload={
        "estimate": psi, "psi": psi, "epsilon": e, "se": se,
        "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "mean_eic": m, "delta": float(delta),
        "max_density_ratio": max(H),
        "method": "TMLE for a stochastic (shift) regime; van der Laan "
                  "& Rose (2018) Chap. 14",
        "note": "the clever covariate is a DENSITY RATIO, not an "
                "inverse probability",
    })


def cheatsheet():
    return ("tlstoch: static and dynamic regimes are both "
            "DETERMINISTIC, and that is the wrong frame twice over -- "
            "you cannot set someone's exercise regime by a rule, and a "
            "media campaign is deterministic at the community level "
            "but stochastic at the individual one; and a deterministic "
            "regime may be UNIDENTIFIABLE because nobody in the data "
            "behaves that way. A stochastic regime SHIFTS the "
            "treatment distribution: Psi = E[Q(A + delta, W)]. "
            "Positivity becomes a support condition -- the shift need "
            "only stay inside the conditional support -- and the "
            "clever covariate is a DENSITY RATIO rather than an "
            "inverse probability.")


# compact alias per ledger/NAMING.md
stochasticregime = shift_tmle
