# morie.fn -- function file (rootcoder007/morie)
r"""Penalised doubly robust TMLE.

In high dimensions the nuisance models must be regularised, and the
regularisation interacts with the targeting in a way that is easy to
get wrong.

**Penalise the nuisance fits, never the targeting step.** The
fluctuation parameter :math:`\epsilon` is one-dimensional and its
maximum likelihood value is what makes the estimator solve
:math:`P_n D^* = 0`. Shrinking :math:`\epsilon` toward zero shrinks the
estimator back toward the untargeted plug-in and breaks the score
equation -- so the penalty belongs on :math:`\bar Q` and :math:`g`,
and the targeting step is left alone. ``penalised_tmle`` enforces that
separation and the anchor demonstrates what shrinking
:math:`\epsilon` costs.

**Post-selection refitting, and why.** The lasso's shrinkage biases the
coefficients of the variables it keeps. Belloni and Chernozhukov's
result is that refitting by ordinary least squares on the *selected*
support removes that shrinkage bias and performs at least as well as
the lasso -- sometimes strictly better -- so "post-lasso" is the
default here rather than the raw lasso fit.

**Why double robustness matters more, not less, under penalisation.**
A penalised nuisance fit is *deliberately* biased. Double robustness
means that bias is tolerable in one nuisance provided the other is
consistent, and the second-order remainder is a product of the two
errors -- so two moderately penalised fits can still yield a
root-:math:`n` estimator, while either one alone could not.

References
----------
Belloni, A. & Chernozhukov, V. (2013) "Least squares after model
selection in high-dimensional sparse models", *Bernoulli* 19(2),
521-547, doi:10.3150/11-BEJ410. Post-lasso: ordinary least squares
applied to the model selected by the lasso removes the shrinkage bias
and performs at least as well as the lasso.

van der Laan, M. J. & Gruber, S. (2016) "One-step targeted minimum
loss-based estimation based on universal least favorable
one-dimensional submodels", *The International Journal of
Biostatistics* 12(1), 351-378, doi:10.1515/ijb-2015-0054. The
one-dimensional fluctuation whose maximum likelihood value must not
itself be penalised.

van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chaps. 4 and 7: the
second-order remainder as a product of nuisance errors, and the rate
conditions under which a penalised initial estimator still yields an
asymptotically linear TMLE.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["lasso_path", "post_lasso", "penalised_tmle",
           "shrunk_targeting_unsafe"]

_EPS = 1e-12


def _logit(p):
    q = min(max(float(p), 1e-9), 1 - 1e-9)
    return math.log(q / (1.0 - q))


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def _soft(x, t):
    return math.copysign(max(abs(x) - t, 0.0), x)


def lasso_path(X, y, lam, iters=500, tol=1e-9):
    r"""Coordinate descent for the L1-penalised least squares fit."""
    rows = [[float(v) for v in r] for r in k.mat(X)]
    t = [float(v) for v in k.vec(y)]
    n, p = len(rows), len(rows[0])
    if len(t) != n:
        raise ValueError("tmldgp: %d rows but %d outcomes"
                         % (n, len(t)))
    if float(lam) < 0.0:
        raise ValueError("tmldgp: lambda cannot be negative")
    b = [0.0] * p
    b0 = sum(t) / n
    for _ in range(int(iters)):
        big = 0.0
        for j in range(p):
            r = [t[i] - b0 - sum(rows[i][q] * b[q]
                                 for q in range(p) if q != j)
                 for i in range(n)]
            zj = sum(rows[i][j] * rows[i][j] for i in range(n))
            if zj < _EPS:
                continue
            new = _soft(sum(rows[i][j] * r[i]
                            for i in range(n)) / n,
                        float(lam)) / (zj / n)
            big = max(big, abs(new - b[j]))
            b[j] = new
        b0 = sum(t[i] - sum(rows[i][q] * b[q] for q in range(p))
                 for i in range(n)) / n
        if big < float(tol):
            break
    return {"beta": b, "intercept": b0,
            "support": [j for j in range(p) if abs(b[j]) > 1e-10],
            "lambda": float(lam)}


def post_lasso(X, y, lam):
    r"""Refit by least squares on the selected support.

    The lasso selects and shrinks; refitting on the selection keeps
    the first and undoes the second.
    """
    rows = [[float(v) for v in r] for r in k.mat(X)]
    t = [float(v) for v in k.vec(y)]
    sel = lasso_path(rows, t, lam)
    S = sel["support"]
    if not S:
        m = sum(t) / len(t)
        return {"support": [], "coef": [], "intercept": m,
                "predict": (lambda row: m), "selected_by": "lasso",
                "note": "the lasso selected nothing"}
    Xs = [[rows[i][j] for j in S] for i in range(len(rows))]
    co = k.wls(Xs, t, [1.0] * len(t), 0.0)["coef"]

    def predict(row):
        v = [float(q) for q in row]
        return co[0] + sum(co[1 + a] * v[S[a]] for a in range(len(S)))

    return {"support": S, "coef": co, "intercept": co[0],
            "predict": predict, "lasso_beta": sel["beta"],
            "selected_by": "lasso, refitted by OLS",
            "note": "post-lasso removes the shrinkage bias on the "
                    "selected coefficients"}


def shrunk_targeting_unsafe(Q, H, Y, ridge=1.0):
    r"""Penalise :math:`\epsilon` itself -- the thing not to do.

    Kept so the cost is measurable: shrinking the fluctuation pulls
    the estimator back toward the untargeted plug-in and leaves
    :math:`P_n D^*` non-zero.
    """
    q = [float(v) for v in k.vec(Q)]
    h = [float(v) for v in k.vec(H)]
    y = [float(v) for v in k.vec(Y)]
    n = len(q)
    off = [_logit(v) for v in q]
    e = 0.0
    for _ in range(60):
        p = [_expit(off[i] + e * h[i]) for i in range(n)]
        gr = sum(h[i] * (y[i] - p[i]) for i in range(n)) \
            - float(ridge) * e
        he = sum(h[i] * h[i] * p[i] * (1 - p[i])
                 for i in range(n)) + float(ridge)
        if he < 1e-12:
            break
        e += gr / he
    upd = [_expit(off[i] + e * h[i]) for i in range(n)]
    return {"epsilon": e, "Q_star": upd,
            "score": sum(h[i] * (y[i] - upd[i])
                         for i in range(n)) / n,
            "caveat": "the score equation is NOT solved when the "
                      "fluctuation is penalised"}


def penalised_tmle(y, D, X, penalty=0.05, iters=100):
    r"""Penalised nuisance fits, unpenalised targeting.

    Both :math:`\bar Q` and :math:`g` are fitted by post-lasso; the
    one-dimensional fluctuation is then fitted by maximum likelihood,
    unregularised.
    """
    yv = [float(v) for v in k.vec(y)]
    a = [float(v) for v in k.vec(D)]
    W = [[float(v) for v in r] for r in k.mat(X)]
    n = len(yv)
    if not (len(a) == len(W) == n):
        raise ValueError("tmldgp: the inputs differ in length")
    if any(v < 0.0 or v > 1.0 for v in yv):
        raise ValueError("tmldgp: the outcome must lie in [0,1]; "
                         "rescale it first (see tmlcou)")
    gfit = post_lasso(W, a, penalty)
    gg = [min(max(float(gfit["predict"](W[i])), 0.02), 0.98)
          for i in range(n)]
    Xa = [[a[i]] + list(W[i]) for i in range(n)]
    qfit = post_lasso(Xa, yv, penalty)
    q1 = [min(max(float(qfit["predict"]([1.0] + list(W[i]))),
                  1e-6), 1 - 1e-6) for i in range(n)]
    q0 = [min(max(float(qfit["predict"]([0.0] + list(W[i]))),
                  1e-6), 1 - 1e-6) for i in range(n)]
    H = [a[i] / gg[i] - (1.0 - a[i]) / (1.0 - gg[i])
         for i in range(n)]
    qa = [q1[i] if a[i] == 1.0 else q0[i] for i in range(n)]
    off = [_logit(v) for v in qa]
    e = 0.0
    for _ in range(int(iters)):
        p = [_expit(off[i] + e * H[i]) for i in range(n)]
        gr = sum(H[i] * (yv[i] - p[i]) for i in range(n))
        he = sum(H[i] * H[i] * p[i] * (1 - p[i]) for i in range(n))
        if he < 1e-12:
            break
        step = gr / he
        e += step
        if abs(step) < 1e-12:
            break
    q1s = [_expit(_logit(q1[i]) + e / gg[i]) for i in range(n)]
    q0s = [_expit(_logit(q0[i]) - e / (1 - gg[i])) for i in range(n)]
    psi = sum(q1s[i] - q0s[i] for i in range(n)) / n
    d = []
    for i in range(n):
        qas = q1s[i] if a[i] == 1.0 else q0s[i]
        d.append(H[i] * (yv[i] - qas) + q1s[i] - q0s[i] - psi)
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / n ** 2)
    return RichResult(payload={
        "estimate": psi, "psi": psi, "epsilon": e, "se": se,
        "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "mean_eic": m, "solves_eic": abs(m) < 1e-6,
        "g_support": gfit["support"], "Q_support": qfit["support"],
        "penalty": float(penalty),
        "method": "penalised doubly robust TMLE with post-lasso "
                  "nuisance fits; Belloni & Chernozhukov (2013), van "
                  "der Laan & Gruber (2016)",
        "note": "the PENALTY is on the nuisances only; penalising the "
                "fluctuation would break the score equation",
    })


def cheatsheet():
    return ("tmldgp: in high dimensions regularise the NUISANCES and "
            "leave the TARGETING alone -- epsilon is one-dimensional "
            "and its MLE is exactly what makes P_n D* = 0, so "
            "shrinking it pulls the estimator back to the untargeted "
            "plug-in. Use POST-LASSO for the nuisance fits: the lasso "
            "selects and shrinks, refitting by OLS on the selection "
            "keeps the selection and undoes the shrinkage. Double "
            "robustness matters MORE under penalisation, since a "
            "penalised fit is deliberately biased and the remainder is "
            "a PRODUCT of the two errors.")


# compact alias per ledger/NAMING.md
penalisedtmle = penalised_tmle
