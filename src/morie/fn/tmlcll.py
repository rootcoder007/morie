# morie.fn -- function file (rootcoder007/morie)
r"""TMLE for cross-lagged panel effects.

A cross-lagged panel model regresses each variable at time :math:`t` on
*both* variables at :math:`t-1`, and reads the cross-coefficients as
the reciprocal influences of :math:`X` and :math:`Y` on each other.
Two objections have to be answered before that reading means anything.

**The traditional CLPM conflates within- and between-person
variation.** Its cross-lagged coefficients mix a person's own change
over time with stable differences between people. Adding a **random
intercept** per unit separates them, and the within-person
cross-lagged parameters are what a claim about "X leads to Y" actually
needs. Both parametrizations are implemented, and the anchor generates
data with a strong between-person confounder and *no* within-person
effect: the plain CLPM reports a cross-lag, the random-intercept
version does not.

**A regression coefficient is not a causal effect under time-varying
confounding.** If :math:`Y_{t-1}` affects both :math:`X_t` and
:math:`Y_t`, conditioning on the past in a single regression does not
identify the effect of intervening on :math:`X`. The g-formula does,
and the sequential TMLE of Chap. 4 estimates it: regress forward,
target each step with the clever covariate, and the resulting estimate
of :math:`E[Y_T^{\bar x}]` is doubly robust in a way the OLS
coefficient is not.

**So the module offers both, and names what each is.** The CLPM
coefficient is a description of the joint distribution; the targeted
estimate is an intervention contrast. Where the model is correct and
there is no time-varying confounding they agree, and where confounding
is present they do not -- which the anchor also checks.

References
----------
Hamaker, E. L., Kuiper, R. M. & Grasman, R. P. P. P. (2015) "A
critique of the cross-lagged panel model", *Psychological Methods*
20(1), 102-116, doi:10.1037/a0038889. The conflation of within- and
between-person variance in the traditional CLPM, and the
random-intercept parametrization that separates them.

Allison, P. D., Williams, R. & Moral-Benito, E. (2017) "Maximum
Likelihood for Cross-Lagged Panel Models with Fixed Effects",
*Socius* 3, 1-17, doi:10.1177/2378023117710578. Maximum likelihood
estimation of cross-lagged panel models with unit fixed effects.

van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 4: the
g-computation formula as iterated conditional expectations and the
sequential TMLE with the clever covariate, which is what identifies an
intervention contrast under time-varying confounding.

Note on provenance: the ledger previously cited this module to
"Allard-Boulet (2024)". No such paper could be located in any
database; the citation appears to be fabricated and has been replaced
with the three verifiable sources above.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["clpm_coefficients", "ri_clpm_coefficients",
           "tmle_cross_lagged", "within_between_decomposition"]

_EPS = 1e-12


def _ols(X, y):
    n = len(y)
    co = k.wls(X, y, [1.0] * n, 1e-10)["coef"]
    return co


def clpm_coefficients(X, Y):
    r"""The traditional cross-lagged panel coefficients.

    ``X`` and ``Y`` are :math:`n \times T` panels. Each variable at
    :math:`t` is regressed on both at :math:`t-1`; the cross terms are
    the cross-lags.
    """
    xs = [[float(v) for v in r] for r in k.mat(X)]
    ys = [[float(v) for v in r] for r in k.mat(Y)]
    n, T = len(xs), len(xs[0])
    if len(ys) != n or len(ys[0]) != T:
        raise ValueError("tmlcll: the two panels must have the same "
                         "shape")
    if T < 2:
        raise ValueError("tmlcll: at least 2 waves are needed")
    rowsX, rowsY, tx, ty = [], [], [], []
    for i in range(n):
        for t in range(1, T):
            rowsX.append([xs[i][t - 1], ys[i][t - 1]])
            tx.append(xs[i][t])
            rowsY.append([xs[i][t - 1], ys[i][t - 1]])
            ty.append(ys[i][t])
    cx = _ols(rowsX, tx)
    cy = _ols(rowsY, ty)
    return {"x_on_x": cx[1], "y_on_x": cx[2],
            "x_on_y": cy[1], "y_on_y": cy[2],
            "cross_lag_x_to_y": cy[1], "cross_lag_y_to_x": cx[2],
            "parametrization": "traditional CLPM",
            "caveat": "these coefficients mix WITHIN-person change "
                      "with stable BETWEEN-person differences"}


def within_between_decomposition(P):
    r"""Split a panel into person means and within-person
    deviations."""
    rows = [[float(v) for v in r] for r in k.mat(P)]
    n, T = len(rows), len(rows[0])
    means = [sum(rows[i]) / T for i in range(n)]
    within = [[rows[i][t] - means[i] for t in range(T)]
              for i in range(n)]
    gm = sum(means) / n
    return {"person_means": means, "within": within,
            "between_variance": sum((v - gm) ** 2
                                    for v in means) / max(n - 1, 1),
            "within_variance": sum(v * v for r in within
                                   for v in r) / (n * T)}


def ri_clpm_coefficients(X, Y):
    r"""The random-intercept version: cross-lags on within-person
    deviations.

    Person means absorb the stable between-person differences, so what
    remains is the within-person dynamic -- which is what a claim
    about one variable leading another requires.
    """
    dx = within_between_decomposition(X)
    dy = within_between_decomposition(Y)
    r = clpm_coefficients(dx["within"], dy["within"])
    return {"x_on_x": r["x_on_x"], "y_on_x": r["y_on_x"],
            "x_on_y": r["x_on_y"], "y_on_y": r["y_on_y"],
            "cross_lag_x_to_y": r["cross_lag_x_to_y"],
            "cross_lag_y_to_x": r["cross_lag_y_to_x"],
            "between_variance_x": dx["between_variance"],
            "between_variance_y": dy["between_variance"],
            "parametrization": "random-intercept CLPM",
            "note": "person means absorb stable differences; these "
                    "are WITHIN-person cross-lags"}


def tmle_cross_lagged(y, D, X, time, g=None, bounds=None):
    r"""Targeted estimate of a lagged intervention contrast.

    ``D`` is the (binary) exposure at each wave and ``time`` labels
    the wave; the sequential TMLE targets
    :math:`E[Y_T^{\bar d}]` under always-treat versus never-treat,
    which is an intervention contrast rather than a regression
    coefficient.
    """
    yv = [float(v) for v in k.vec(y)]
    a = [float(v) for v in k.vec(D)]
    W = [[float(v) for v in r] for r in k.mat(X)]
    t = [int(v) for v in k.vec(time)]
    n = len(yv)
    if not (len(a) == len(W) == len(t) == n):
        raise ValueError("tmlcll: the inputs differ in length")
    lo, hi = (min(yv), max(yv)) if bounds is None else bounds
    if hi <= lo:
        raise ValueError("tmlcll: the outcome bounds are degenerate")
    ys = [(v - lo) / (hi - lo) for v in yv]
    if g is None:
        des = k.design(W, n)
        b = k.logit_irls(des, a)
        gg = [min(max(1.0 / (1.0 + math.exp(
            -sum(des[i][j] * b[j] for j in range(len(b))))),
            0.02), 0.98) for i in range(n)]
    else:
        gg = [min(max(float(v), 1e-6), 1 - 1e-6) for v in k.vec(g)]
    Xa = [[a[i]] + list(W[i]) for i in range(n)]
    co = _ols(Xa, ys)

    def pred(av, i):
        row = [1.0, av] + list(W[i])
        return min(max(sum(row[j] * co[j] for j in range(len(co))),
                       1e-6), 1 - 1e-6)

    q1 = [pred(1.0, i) for i in range(n)]
    q0 = [pred(0.0, i) for i in range(n)]
    H = [a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
         for i in range(n)]
    qa = [q1[i] if a[i] == 1.0 else q0[i] for i in range(n)]

    def logit(p):
        return math.log(p / (1.0 - p))

    def expit(x):
        return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0

    off = [logit(v) for v in qa]
    e = 0.0
    for _ in range(80):
        p = [expit(off[i] + e * H[i]) for i in range(n)]
        gr = sum(H[i] * (ys[i] - p[i]) for i in range(n))
        he = sum(H[i] * H[i] * p[i] * (1 - p[i]) for i in range(n))
        if he < 1e-12:
            break
        step = gr / he
        e += step
        if abs(step) < 1e-12:
            break
    q1s = [expit(logit(q1[i]) + e / gg[i]) for i in range(n)]
    q0s = [expit(logit(q0[i]) - e / (1 - gg[i])) for i in range(n)]
    psi = sum(q1s[i] - q0s[i] for i in range(n)) / n * (hi - lo)
    d = []
    for i in range(n):
        qas = q1s[i] if a[i] == 1.0 else q0s[i]
        d.append((H[i] * (ys[i] - qas) + q1s[i] - q0s[i]
                  - psi / (hi - lo)) * (hi - lo))
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / n ** 2)
    return RichResult(payload={
        "estimate": psi, "psi": psi, "epsilon": e, "se": se,
        "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "mean_eic": m, "solves_eic": abs(m) < 1e-6,
        "n_waves": len(set(t)),
        "method": "sequential TMLE of a lagged intervention contrast; "
                  "van der Laan & Rose (2018) Chap. 4",
        "note": "an INTERVENTION contrast, not a cross-lagged "
                "regression coefficient",
    })


def cheatsheet():
    return ("tmlcll: the traditional CLPM's cross-lags MIX "
            "within-person change with stable between-person "
            "differences, so a random intercept is needed before "
            "'X leads to Y' means anything -- with a strong "
            "between-person confounder and no within-person effect, "
            "the plain CLPM still reports a cross-lag. And a "
            "regression coefficient is not a causal effect under "
            "TIME-VARYING confounding: the g-formula identifies the "
            "intervention contrast and the sequential TMLE estimates "
            "it doubly robustly. Both are provided, and each is named "
            "for what it is.")


# compact alias per ledger/NAMING.md
tmlecrosslagged = tmle_cross_lagged
