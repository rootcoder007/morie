# morie.fn -- function file (rootcoder007/morie)
r"""LTMLE: targeting the sequential regressions.

The g-computation estimand can be written as iterated conditional
expectations: regress the outcome on the history at the last time
point, evaluate that fit under the intervention rule, treat the result
as the outcome one step earlier, and repeat. Estimating those
regressions with machine learning gives a substitution estimator that
is *not* asymptotically linear -- the bias of a data-adaptive fit does
not vanish fast enough. Targeting fixes precisely that.

**Each step is a one-dimensional fluctuation.** Since the outcome
regressions are bounded in :math:`[0,1]` (after scaling), the loss is
the Bernoulli log-likelihood and the submodel is logistic with the
initial fit as **offset**:

.. math:: \mathrm{logit}\,\bar Q_t(\epsilon) =
          \mathrm{logit}\,\bar Q_t^0 + \epsilon\, H_t,

where the **clever covariate** :math:`H_t` is the inverse cumulative
probability of following the rule through time :math:`t`,

.. math:: H_t = \frac{\prod_{s \le t}
          I(A_s = d_s)}{\prod_{s \le t} g_s}.

Fitting :math:`\epsilon` by maximum likelihood makes the updated fit
solve the efficient influence curve equation for that component; doing
it at every time point, backwards, makes the whole estimator solve
:math:`P_n D^* = 0`.

**Double robustness, stated exactly.** The estimator is consistent if
*either* the sequential outcome regressions *or* the treatment
mechanism are consistently estimated -- not both. The anchor exploits
that: it breaks each arm separately and requires the estimate to
survive, then breaks both and requires it to fail. Two wrong arms are
the case that must not silently pass.

**Positivity is the binding constraint.** The clever covariate is an
inverse probability; as the cumulative probability of the rule
approaches zero it explodes, and the second-order remainder is bounded
only when :math:`g_{0} > \delta > 0`. That is why the module reports
the largest clever covariate rather than hiding it.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 4 (the
g-computation formula as iterated conditional expectations; the
efficient influence curve of the longitudinal parameter; the
Bernoulli log-likelihood loss and the logistic submodel through the
initial estimator with the clever covariate; the sequential definition
of loss and submodel for each Q_t; the second-order remainder and the
positivity condition g > delta > 0 that bounds it; and the use of a
super learner containing the highly adaptive lasso as the initial
estimator). Chap. 3 (the sequential regressions being targeted).

van der Laan, M. J. & Gruber, S. (2012) "Targeted minimum loss based
estimation of causal effects of multiple time point interventions",
*International Journal of Biostatistics* 8(1), Article 9,
doi:10.1515/1557-4679.1370.

Bang, H. & Robins, J. M. (2005) "Doubly robust estimation in missing
data and causal inference models", *Biometrics* 61(4), 962-973,
doi:10.1111/j.1541-0420.2005.00377.x.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["clever_covariate", "fluctuate", "tmle_point",
           "ltmle", "influence_curve_se"]

_EPS = 1e-12


def _logit(p):
    q = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(q / (1.0 - q))


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def clever_covariate(A, g, rule=1.0):
    r""":math:`H = I(A = d)/g`, the inverse probability of following
    the rule.

    Reported with its maximum, because a large clever covariate is a
    positivity problem rather than a large number.
    """
    a = [float(v) for v in k.vec(A)]
    gg = [float(v) for v in k.vec(g)]
    if len(a) != len(gg):
        raise ValueError("ltmle: %d treatments but %d propensities"
                         % (len(a), len(gg)))
    if any(v <= 0.0 or v >= 1.0 for v in gg):
        raise ValueError("ltmle: propensities must lie strictly "
                         "inside (0,1)")
    h = [(1.0 if a[i] == float(rule) else 0.0) / gg[i]
         for i in range(len(a))]
    return {"H": h, "max": max(h), "mean": sum(h) / len(h),
            "note": "a large clever covariate IS the positivity "
                    "violation showing itself"}


def fluctuate(Q, H, Y, iters=100, tol=1e-10):
    r"""Fit :math:`\epsilon` in the logistic submodel by Newton steps.

    The initial fit enters as an offset, which is what makes the
    update a *fluctuation* of it rather than a refit.
    """
    q = [float(v) for v in k.vec(Q)]
    h = [float(v) for v in k.vec(H)]
    y = [float(v) for v in k.vec(Y)]
    n = len(q)
    if not (len(h) == len(y) == n):
        raise ValueError("ltmle: the inputs differ in length")
    off = [_logit(v) for v in q]
    eps = 0.0
    for _ in range(int(iters)):
        p = [_expit(off[i] + eps * h[i]) for i in range(n)]
        gr = sum(h[i] * (y[i] - p[i]) for i in range(n))
        he = sum(h[i] * h[i] * p[i] * (1.0 - p[i]) for i in range(n))
        if he < 1e-12:
            break
        step = gr / he
        eps += step
        if abs(step) < float(tol):
            break
    upd = [_expit(off[i] + eps * h[i]) for i in range(n)]
    return {"epsilon": eps, "Q_star": upd,
            "score": sum(h[i] * (y[i] - upd[i])
                         for i in range(n)) / n}


def tmle_point(A, Y, Q1, Q0, g):
    r"""Point-treatment TMLE of the ATE.

    Both arms are fluctuated with their own clever covariate, then the
    substitution estimator is computed from the updated fits.
    """
    a = [float(v) for v in k.vec(A)]
    y = [float(v) for v in k.vec(Y)]
    q1 = [float(v) for v in k.vec(Q1)]
    q0 = [float(v) for v in k.vec(Q0)]
    gg = [float(v) for v in k.vec(g)]
    n = len(a)
    H = [a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
         for i in range(n)]
    qa = [q1[i] if a[i] == 1.0 else q0[i] for i in range(n)]
    fl = fluctuate(qa, H, y)
    e = fl["epsilon"]
    q1s = [_expit(_logit(q1[i]) + e * (1.0 / gg[i]))
           for i in range(n)]
    q0s = [_expit(_logit(q0[i]) - e * (1.0 / (1.0 - gg[i])))
           for i in range(n)]
    psi = sum(q1s[i] - q0s[i] for i in range(n)) / n
    d = []
    for i in range(n):
        qas = q1s[i] if a[i] == 1.0 else q0s[i]
        d.append(H[i] * (y[i] - qas) + q1s[i] - q0s[i] - psi)
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / n ** 2)
    return RichResult(payload={
        "estimate": psi, "psi": psi, "epsilon": e,
        "se": se, "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "mean_eic": m, "solves_eic": abs(m) < 1e-6,
        "max_clever_covariate": max(abs(v) for v in H),
        "initial_plugin": sum(q1[i] - q0[i] for i in range(n)) / n,
        "method": "TMLE with a logistic submodel and clever "
                  "covariate; van der Laan & Rose (2018) Chap. 4",
        "note": "consistent if EITHER the outcome regression OR the "
                "treatment mechanism is consistent",
    })


def ltmle(Q_seq, H_seq, Y_seq):
    r"""Longitudinal TMLE: fluctuate backwards through the sequence.

    ``Q_seq[t]`` is the initial fit at time :math:`t`, ``H_seq[t]``
    its clever covariate, and the updated fit at :math:`t+1` becomes
    the outcome regressed at :math:`t`.
    """
    T = len(Q_seq)
    if T < 1:
        raise ValueError("ltmle: the sequence is empty")
    if len(H_seq) != T:
        raise ValueError("ltmle: %d fits but %d clever covariates"
                         % (T, len(H_seq)))
    eps, current = [], [float(v) for v in k.vec(Y_seq[-1])]
    stars = []
    for t in range(T - 1, -1, -1):
        fl = fluctuate(Q_seq[t], H_seq[t], current)
        eps.append(fl["epsilon"])
        current = fl["Q_star"]
        stars.append(current)
    psi = sum(current) / len(current)
    return RichResult(payload={
        "estimate": psi, "psi": psi,
        "epsilons": list(reversed(eps)),
        "Q_star": list(reversed(stars)), "T": T,
        "method": "LTMLE by backward sequential fluctuation; van der "
                  "Laan & Rose (2018) Chap. 4",
    })


def influence_curve_se(d):
    r""":math:`\sqrt{\mathrm{var}(D^*)/n}`."""
    v = [float(q) for q in k.vec(d)]
    n = len(v)
    if n < 2:
        raise ValueError("ltmle: at least 2 observations are needed")
    m = sum(v) / n
    return math.sqrt(sum((q - m) ** 2 for q in v) / (n - 1) / n)


def cheatsheet():
    return ("tlltmle: write the g-formula as ITERATED conditional "
            "expectations, fit them with machine learning, then TARGET "
            "each one. Every step is a one-dimensional logistic "
            "fluctuation with the initial fit as OFFSET and the clever "
            "covariate H = I(A = d)/g as the covariate; the MLE for "
            "epsilon makes the update solve the efficient influence "
            "curve equation. DOUBLE ROBUST: consistent if EITHER the "
            "outcome regressions OR the treatment mechanism is right "
            "-- not both. The clever covariate is an inverse "
            "probability, so a large one IS the positivity violation.")


# compact alias per ledger/NAMING.md
longitudinaltmle = ltmle
