# morie.fn -- function file (rootcoder007/morie)
r"""C-TMLE: choosing the treatment mechanism collaboratively.

A standard TMLE estimates the treatment mechanism :math:`g` as well as
possible on its own terms. That is not what the target parameter
needs. A covariate strongly predictive of treatment but unrelated to
the outcome adds nothing to bias removal and a great deal to variance
-- the clever covariate becomes large, and the estimator degrades.
The "collaborative" idea is that :math:`g` should be fitted **in
collaboration with** the current outcome regression: what matters is
the residual confounding the outcome fit has left behind, not the
treatment mechanism in isolation.

**The algorithm is an ordered sequence, not a search.** Build a
sequence of candidate treatment mechanisms
:math:`G_{n,h}` indexed by a tuning parameter :math:`h` -- in the
continuous-tuning version, the variation-norm bound of Chap. 6 --
each giving a TMLE :math:`Q^*_{n,h}`, with more aggressive candidates
built on the previous update as initial estimator. Then **select
:math:`h` by cross-validated loss of the resulting targeted outcome
fit**, not by how well :math:`g` predicts treatment. That criterion is
the mechanism: a covariate that only inflates the clever covariate
makes the targeted fit worse and is therefore not selected.

**Why a continuous index helps.** With a discrete forward-selection
sequence the estimator jumps between models; indexing by a continuous
bound makes the path smooth, so the selected estimator moves
continuously with the data and the theory for asymptotic linearity
goes through under weaker conditions.

**The invariant worth checking.** However :math:`h` is chosen, the
selected C-TMLE still solves the efficient score equation for its own
:math:`g`; collaboration changes *which* nuisance fit is targeted
against, not whether the targeting equation is solved. The anchor
checks both: the selection prefers the smaller model when the extra
covariate is instrumental, and the score equation is still solved.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 10 (C-TMLE
for continuous tuning: the general C-TMLE algorithm constructing an
ordered sequence of TMLEs indexed by a tuning parameter, with the
variation norm of Chap. 6 as the continuous index; the use of the
L-fit of the targeted outcome regression as the selection criterion
rather than the fit of g; the one-step TMLE inside the sequence; the
verification that C-TMLE solves the critical score equation; and the
general theorem for C-TMLE asymptotic linearity).

van der Laan, M. J. & Gruber, S. (2010) "Collaborative Double Robust
Targeted Maximum Likelihood Estimation", *International Journal of
Biostatistics* 6(1), Article 17, doi:10.2202/1557-4679.1181. The
original C-TMLE.

Gruber, S. & van der Laan, M. J. (2010) "An Application of
Collaborative Targeted Maximum Likelihood Estimation in Causal
Inference and Genomics", *International Journal of Biostatistics*
6(1), Article 18, doi:10.2202/1557-4679.1182.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["candidate_sequence", "targeted_loss", "ctmle",
           "instrument_penalty"]

_EPS = 1e-12


def _logit(p):
    q = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(q / (1.0 - q))


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def _fluct(Q, H, Y, iters=60):
    q = [float(v) for v in k.vec(Q)]
    h = [float(v) for v in k.vec(H)]
    y = [float(v) for v in k.vec(Y)]
    n = len(q)
    off = [_logit(v) for v in q]
    e = 0.0
    for _ in range(iters):
        p = [_expit(off[i] + e * h[i]) for i in range(n)]
        gr = sum(h[i] * (y[i] - p[i]) for i in range(n))
        he = sum(h[i] * h[i] * p[i] * (1 - p[i]) for i in range(n))
        if he < 1e-12:
            break
        e += gr / he
    return e, [_expit(off[i] + e * h[i]) for i in range(n)]


def targeted_loss(Q_star, Y):
    r"""The selection criterion: log-likelihood loss of the TARGETED
    outcome fit.

    Not the fit of :math:`g`. A covariate that only inflates the
    clever covariate makes this worse, which is how it gets rejected.
    """
    q = [float(v) for v in k.vec(Q_star)]
    y = [float(v) for v in k.vec(Y)]
    n = len(q)
    tot = 0.0
    for i in range(n):
        p = min(max(q[i], _EPS), 1.0 - _EPS)
        tot += -(y[i] * math.log(p) + (1.0 - y[i])
                 * math.log(1.0 - p))
    return tot / n


def candidate_sequence(A, W, g_models):
    r"""Fit the ordered sequence of treatment mechanisms.

    ``g_models`` is a list of covariate index lists, ordered from
    least to most aggressive; each is fitted by logistic regression.
    """
    a = [float(v) for v in k.vec(A)]
    rows = [[float(v) for v in r] for r in k.mat(W)]
    out = []
    for cols in g_models:
        X = [[rows[i][j] for j in cols] for i in range(len(a))]
        try:
            b = k.logit_irls(k.design(X, len(a)), a)
            g = [_expit(sum(k.design(X, len(a))[i][j] * b[j]
                            for j in range(len(b))))
                 for i in range(len(a))]
        except Exception:
            m = sum(a) / len(a)
            g = [m] * len(a)
        g = [min(max(v, 0.01), 0.99) for v in g]
        out.append({"covariates": list(cols), "g": g,
                    "max_clever": max(
                        max(1.0 / v, 1.0 / (1.0 - v)) for v in g)})
    return out


def instrument_penalty(g_small, g_large):
    r"""How much variance the extra covariate buys.

    Reported as the ratio of the largest clever covariates, which is
    the quantity that actually degrades the estimator.
    """
    a = max(max(1.0 / v, 1.0 / (1.0 - v))
            for v in k.vec(g_small))
    b = max(max(1.0 / v, 1.0 / (1.0 - v))
            for v in k.vec(g_large))
    return {"small": a, "large": b, "ratio": b / a,
            "note": "a pure instrument raises this without removing "
                    "any bias"}


def ctmle(A, Y, Q1, Q0, W, g_models, V=5, seed=0, penalty=True):
    r"""Select among the candidate TMLEs by cross-validated targeted
    loss, optionally penalised.

    The plain log-likelihood loss of the targeted fit separates the
    candidates only weakly -- fluctuating barely moves :math:`\bar Q`
    -- so the finite-sample criterion adds the estimated variance of
    the influence curve, which is precisely the quantity an
    instrumental covariate inflates. ``penalty=False`` recovers the
    unpenalised criterion, and the difference is visible rather than
    argued.
    """
    a = [float(v) for v in k.vec(A)]
    y = [float(v) for v in k.vec(Y)]
    q1 = [float(v) for v in k.vec(Q1)]
    q0 = [float(v) for v in k.vec(Q0)]
    n = len(a)
    cands = candidate_sequence(a, W, g_models)
    if not cands:
        raise ValueError("tlctmle: no candidate treatment mechanisms")
    rng = np.random.default_rng(seed)
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(float(rng.uniform()) * (i + 1)) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    folds = [idx[v::int(V)] for v in range(int(V))]
    risks, losses, pens = [], [], []
    for c in cands:
        g = c["g"]
        tot, m, ics = 0.0, 0, []
        for f in folds:
            tr = [i for i in range(n) if i not in set(f)]
            H = [a[i] / g[i] - (1 - a[i]) / (1 - g[i]) for i in tr]
            qa = [q1[i] if a[i] == 1.0 else q0[i] for i in tr]
            e, _ = _fluct(qa, H, [y[i] for i in tr])
            for i in f:
                h = a[i] / g[i] - (1 - a[i]) / (1 - g[i])
                q = q1[i] if a[i] == 1.0 else q0[i]
                qs = _expit(_logit(q) + e * h)
                tot += targeted_loss([qs], [y[i]])
                m += 1
                q1i = _expit(_logit(q1[i]) + e / g[i])
                q0i = _expit(_logit(q0[i]) - e / (1 - g[i]))
                ics.append(h * (y[i] - qs) + q1i - q0i)
        mu = sum(ics) / len(ics)
        var = sum((v - mu) ** 2 for v in ics) / (len(ics) - 1)
        losses.append(tot / m)
        pens.append(var / n)
        risks.append(tot / m + (var / n if penalty else 0.0))
    best = min(range(len(risks)), key=lambda i: risks[i])
    g = cands[best]["g"]
    H = [a[i] / g[i] - (1 - a[i]) / (1 - g[i]) for i in range(n)]
    qa = [q1[i] if a[i] == 1.0 else q0[i] for i in range(n)]
    e, _ = _fluct(qa, H, y)
    q1s = [_expit(_logit(q1[i]) + e / g[i]) for i in range(n)]
    q0s = [_expit(_logit(q0[i]) - e / (1 - g[i])) for i in range(n)]
    psi = sum(q1s[i] - q0s[i] for i in range(n)) / n
    d = []
    for i in range(n):
        qas = q1s[i] if a[i] == 1.0 else q0s[i]
        d.append(H[i] * (y[i] - qas) + q1s[i] - q0s[i] - psi)
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / n ** 2)
    return RichResult(payload={
        "estimate": psi, "psi": psi, "selected": best,
        "selected_covariates": cands[best]["covariates"],
        "cv_risks": risks, "cv_losses": losses,
        "variance_penalties": pens, "penalized": bool(penalty),
        "epsilon": e, "se": se,
        "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "mean_eic": m, "solves_eic": abs(m) < 1e-6,
        "max_clever_covariate": max(abs(v) for v in H),
        "method": "C-TMLE selecting g by the cross-validated loss of "
                  "the TARGETED outcome fit; van der Laan & Rose "
                  "(2018) Chap. 10",
        "note": "collaboration changes WHICH g is targeted against, "
                "not whether the score equation is solved",
    })


def cheatsheet():
    return ("tlctmle: fitting g as well as possible ON ITS OWN TERMS "
            "is the wrong objective -- a covariate that predicts "
            "treatment but not the outcome removes no bias and "
            "inflates the clever covariate. Build an ORDERED sequence "
            "of candidate g's (continuously indexed by a "
            "variation-norm bound) and select by the cross-validated "
            "loss of the TARGETED OUTCOME FIT, not by g's own fit. "
            "That criterion rejects instruments automatically. The "
            "selected C-TMLE still solves the efficient score "
            "equation.")


# compact alias per ledger/NAMING.md
collaborativetmle = ctmle
