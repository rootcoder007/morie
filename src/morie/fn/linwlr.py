# morie.fn -- function file (rootcoder007/morie)
r"""Weighted linear learner for a structural nested mean model.

Robins (2004) develops optimal structural nested models for sequential
decisions. The object it estimates is the **blip function**

.. math:: \gamma(a, w) = E\left[Y^{a} - Y^{0} \mid A = a, W = w\right],

the effect of receiving :math:`a` rather than nothing among those who
received :math:`a` with history :math:`w`. For a linear blip
:math:`\gamma(a, w) = a\,(\psi_0 + \psi_1^{\top} w)`, g-estimation
solves the score equation built from the residual

.. math:: H(\psi) = Y - a\,(\psi_0 + \psi_1^{\top} w),

which is the outcome with the treatment effect *removed*. Under the
model, :math:`H(\psi)` is independent of :math:`A` given :math:`W`, so
:math:`\psi` is the value at which the association between
:math:`H(\psi)` and :math:`A - E[A \mid W]` vanishes. Because the blip
is linear this has a closed form and no search is needed -- which is
also why it is called a *learner*: the same fit is what the A-learning
literature calls the contrast model.

Two routes, both here:

``"gest"`` (default)
    G-estimation on the centred treatment :math:`A - \pi(W)`. Consistent
    if the **propensity** model is right, whatever the outcome model
    does, because the centring makes the estimating function have mean
    zero under any outcome surface.
``"wls"``
    Weighted least squares of Y on the blip basis with IP weights. The
    route the module's name points at, consistent if the propensity is
    right, and the two agree exactly when the blip basis and the
    propensity covariates coincide -- which the anchor checks.

**The double-robustness claim people expect is not made here.** A
doubly robust g-estimator needs an outcome model as well, and adding a
misspecified one silently would be worse than not having it. What is
implemented is the propensity-consistent estimator; ``baseline`` may be
supplied to residualise Y first, which improves efficiency without
changing what has to be right.

References
----------
Robins, J. M. (2004) "Optimal structural nested models for optimal
sequential decisions", in Lin, D. Y. & Heagerty, P. J. (eds.),
*Proceedings of the Second Seattle Symposium in Biostatistics*, Lecture
Notes in Statistics 179, Springer, 189-326,
doi:10.1007/978-1-4419-9076-1_11.

Robins, J. M. (1994) "Correcting for non-compliance in randomized
trials using structural nested mean models", *Communications in
Statistics -- Theory and Methods* 23(8), 2379-2412,
doi:10.1080/03610929408831393 -- the structural nested mean model and
g-estimation in their original form.

Hernan, M. A. & Robins, J. M. (2020) *Causal Inference: What If*,
Chapman & Hall/CRC, Ch. 14 (g-estimation of structural nested models).
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["linear_weighted_learner", "blip"]

_METHODS = ("gest", "wls")


def blip(a, w, psi):
    """gamma(a, w) = a (psi_0 + psi_1' w), the linear blip."""
    av = k.vec(a)
    Wm = k.mat(w) if w is not None else [[]] * len(av)
    out = []
    for i in range(len(av)):
        s = psi[0] + sum(psi[j + 1] * Wm[i][j] for j in range(len(psi) - 1))
        out.append(av[i] * s)
    return out


def linear_weighted_learner(y, A, W, propensity=None, method="gest",
                            baseline=None, pi_covariates=None,
                            ridge=1e-10):
    r"""Estimate a linear blip function.

    Parameters
    ----------
    y : array-like
        Outcome.
    A : array-like
        Treatment. Binary or continuous; the blip is linear in it.
    W : array-like
        The history the blip is allowed to depend on, n-by-p. Pass None
        for a constant blip.
    propensity : array-like, optional
        E[A | W]. Fitted by logistic regression for a binary treatment,
        or linear regression otherwise, when not supplied.
    pi_covariates : array-like, optional
        Covariates for the propensity model. Defaults to `W`, but the
        roles are genuinely different: `W` is what the EFFECT may depend
        on, while the propensity must condition on everything that
        confounds treatment. A constant blip with confounders needs
        `W=None` and `pi_covariates=L`; getting it wrong is silent, and
        the estimator simply returns a confounded number.
    method : {"gest", "wls"}
    baseline : array-like, optional
        Covariates to residualise Y against first, for efficiency. Does
        not change what must be correctly specified.

    Returns
    -------
    RichResult
        ``estimate`` is psi_0, the blip intercept -- the effect at
        W = 0. ``psi`` is the whole vector.

    Examples
    --------
    A constant blip with a true effect of 2::

        r = linear_weighted_learner(y, A, None, pi_covariates=L)
        r["estimate"]
    """
    if method not in _METHODS:
        raise ValueError("linear_weighted_learner: method must be 'gest' "
                         "or 'wls', got %r" % (method,))
    yv = k.vec(y)
    av = k.vec(A)
    n = len(yv)
    if len(av) != n:
        raise ValueError("linear_weighted_learner: %d outcomes but %d "
                         "treatments" % (n, len(av)))
    Wm = k.mat(W) if W is not None else [[] for _ in range(n)]
    if len(Wm) != n:
        raise ValueError("linear_weighted_learner: %d outcomes but %d "
                         "history rows" % (n, len(Wm)))
    binary = all(v in (0.0, 1.0) for v in av)

    if propensity is None:
        if pi_covariates is not None:
            Zsrc = k.mat(pi_covariates)
            if len(Zsrc) != n:
                raise ValueError(
                    "linear_weighted_learner: %d propensity covariate "
                    "rows for %d observations" % (len(Zsrc), n))
        else:
            Zsrc = Wm if Wm and Wm[0] else None
        Z = k.design(Zsrc, n)
        if binary:
            pi = [k.sigmoid(v) for v in k.matvec(Z, k.logit_irls(Z, av, 60,
                                                                 ridge))]
        else:
            pi = k.matvec(Z, k.lstsq(Z, av, ridge))
    else:
        pi = [float(v) for v in k.vec(propensity)]
        if len(pi) != n:
            raise ValueError("linear_weighted_learner: %d propensities "
                             "for %d observations" % (len(pi), n))
    if binary and any(p <= 0.0 or p >= 1.0 for p in pi):
        raise ValueError("linear_weighted_learner: a propensity of 0 or 1 "
                         "violates positivity and makes the blip "
                         "unidentified there")

    ytilde = list(yv)
    if baseline is not None:
        Zb = k.design(k.mat(baseline), n)
        bb = k.lstsq(Zb, yv, ridge)
        fitted = k.matvec(Zb, bb)
        ytilde = [yv[i] - fitted[i] for i in range(n)]

    p = len(Wm[0]) if Wm and Wm[0] else 0
    if method == "gest":
        # Score: sum_i (A_i - pi_i) [1, W_i] (Y_i - A_i [1,W_i]' psi) = 0
        # which is linear in psi, so it solves in closed form.
        basis = [[1.0] + list(Wm[i]) for i in range(n)]
        cen = [av[i] - pi[i] for i in range(n)]
        q = p + 1
        M = [[sum(cen[i] * basis[i][a] * av[i] * basis[i][b]
                  for i in range(n)) for b in range(q)] for a in range(q)]
        rhs = [sum(cen[i] * basis[i][a] * ytilde[i] for i in range(n))
               for a in range(q)]
        for a in range(q):
            M[a][a] += ridge
        psi = k.ridgesolve(M, rhs, ridge)
        resid = [ytilde[i] - av[i] * sum(psi[j] * basis[i][j]
                                         for j in range(q))
                 for i in range(n)]
        # sandwich variance for the linear estimating equation
        bread = [[sum(cen[i] * basis[i][a] * av[i] * basis[i][b]
                      for i in range(n)) / n for b in range(q)]
                 for a in range(q)]
        meat = [[sum((cen[i] * basis[i][a] * resid[i])
                     * (cen[i] * basis[i][b] * resid[i])
                     for i in range(n)) / n for b in range(q)]
                for a in range(q)]
        se = _sandwich_se(bread, meat, n, ridge)
    else:
        w = [1.0 / pi[i] if av[i] > 0.5 else 1.0 / (1.0 - pi[i])
             for i in range(n)] if binary else [1.0] * n
        X = [[av[i]] + [av[i] * Wm[i][j] for j in range(p)]
             + list(Wm[i]) for i in range(n)]
        fit = k.wls(X, ytilde, w)
        psi = [fit["coef"][1]] + [fit["coef"][2 + j] for j in range(p)]
        se = [fit["se"][1]] + [fit["se"][2 + j] for j in range(p)]
        resid = fit["resid"]

    return RichResult(payload={
        "estimate": psi[0], "se": se[0] if se else float("nan"),
        "psi": psi, "psi_se": se,
        "propensity": pi, "residual": resid,
        "blip": blip(av, Wm if p else None, psi),
        "binary_treatment": binary, "method_used": method, "n": n,
        "method": "linear blip by %s, Robins (2004) optimal structural "
                  "nested models" % ("g-estimation" if method == "gest"
                                     else "weighted least squares"),
    })


def _sandwich_se(bread, meat, n, ridge):
    q = len(bread)
    inv = [k.ridgesolve(bread, [1.0 if j == a else 0.0 for j in range(q)],
                        ridge) for a in range(q)]
    out = []
    for a in range(q):
        t = 0.0
        for u in range(q):
            for v in range(q):
                t += inv[a][u] * meat[u][v] * inv[a][v]
        out.append((t / n) ** 0.5 if t > 0.0 else float("nan"))
    return out


def cheatsheet():
    return ("linwlr: linear blip gamma(a,w) = a(psi0 + psi1'w) by "
            "g-estimation on A - E[A|W] (Robins 2004), or by IP-weighted "
            "least squares. Consistent if the PROPENSITY is right; no "
            "double-robustness claimed without an outcome model.")


# compact alias per ledger/NAMING.md
linearweightedlearner = linear_weighted_learner
