# morie.fn -- function file (rootcoder007/morie)
r"""The highly adaptive lasso.

Most flexible estimators buy their rate with a smoothness assumption.
HAL replaces smoothness with a **variation norm** bound, which is a far
weaker demand: the target need only be cadlag (right-continuous with
left limits) with finite variation norm -- it may jump, and it need not
be differentiable anywhere.

**Representation.** Any such :math:`d`-variate function can be written
as a sum over subsets of its sections against indicator basis
functions,

.. math:: h(x) = h(0) + \sum_{S \subset \{1,\dots,d\}}
          \int \prod_{j \in S} I(x_j \ge u_j)\, dh_S(u_S),

so approximating :math:`h` by a **linear combination of indicator
basis functions** loses nothing in the limit. Discretising the
measures makes the coefficients a finite vector whose
:math:`\ell_1` norm *is* the variation norm of the approximation --
that identity is why the lasso constraint is the right constraint,
rather than a convenient one.

**The estimator.** Minimise the empirical risk over such linear
combinations subject to

.. math:: \sum_j |\beta_j| \le \lambda,

with :math:`\lambda` chosen by cross-validation. So HAL is a lasso in
a very high-dimensional basis, and its tuning parameter has a direct
interpretation as a bound on the variation norm of the fit.

**The rate is the point.** Under only the cadlag and finite-variation
conditions, and *including* fully nonparametric models and
high-dimensional data, HAL converges at a rate faster than
:math:`n^{-1/4}` -- specifically :math:`n^{-1/3}` up to a log factor.
That threshold is not arbitrary: :math:`n^{-1/4}` is exactly what
double robust efficiency arguments require of the nuisance estimators,
so an estimator that clears it makes the TMLE built on it efficient
under weak conditions.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 6 (the
assumption that the target is cadlag with finite variation norm; the
representation of a d-variate cadlag function as a sum over subsets of
integrals against products of indicator basis functions; the
definition of the variation norm as the sum of the variation norms of
the sections; the estimator as the minimiser of empirical risk over
linear combinations of indicator basis functions subject to the sum of
absolute coefficients being bounded by lambda, itself selected by
cross-validation; the identity between that L1 bound and the variation
norm of the discrete approximation; and the guarantee of a rate faster
than n^{-1/4} even for complete nonparametric models and
high-dimensional data structures).

van der Laan, M. J. (2017) "A generally efficient targeted minimum
loss based estimator based on the highly adaptive lasso",
*International Journal of Biostatistics* 13(2), 20150097,
doi:10.1515/ijb-2015-0097.

Benkeser, D. & van der Laan, M. J. (2016) "The Highly Adaptive
Lasso Estimator", *Proceedings of the 2016 IEEE International
Conference on Data Science and Advanced Analytics (DSAA)*, 689-696,
doi:10.1109/DSAA.2016.93.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["indicator_basis", "variation_norm", "hal_fit",
           "hal_predict", "cv_select_lambda"]

_EPS = 1e-12


def indicator_basis(X, knots=None, max_order=2):
    r"""The design of products :math:`\prod_{j\in S} I(x_j \ge u_j)`.

    ``max_order`` bounds :math:`|S|`; order 1 gives main terms, order
    2 adds every pairwise interaction, and so on up to :math:`2^d`.
    """
    rows = [[float(v) for v in r] for r in k.mat(X)]
    n, d = len(rows), len(rows[0])
    K = rows if knots is None else [[float(v) for v in r]
                                    for r in k.mat(knots)]
    subsets = [[j] for j in range(d)]
    if int(max_order) >= 2:
        subsets += [[a, b] for a in range(d) for b in range(a + 1, d)]
    if int(max_order) >= 3:
        subsets += [[a, b, c] for a in range(d)
                    for b in range(a + 1, d)
                    for c in range(b + 1, d)]
    cols = []
    for S in subsets:
        for u in K:
            cols.append((tuple(S), tuple(u[j] for j in S)))
    design = []
    for i in range(n):
        design.append([1.0 if all(rows[i][S[t]] >= v[t]
                                  for t in range(len(S)))
                       else 0.0 for (S, v) in cols])
    return {"design": design, "columns": cols,
            "n_basis": len(cols), "max_order": int(max_order)}


def variation_norm(beta):
    r"""The variation norm of the fit: :math:`\sum_j |\beta_j|`.

    Not an analogy -- for the discrete approximation the two are the
    same number.
    """
    b = [float(v) for v in k.vec(beta)]
    return sum(abs(v) for v in b)


def hal_fit(X, y, lam=1.0, iters=2000, step=0.05, max_order=2,
            knots=None, intercept=True):
    r"""Empirical risk minimisation under
    :math:`\sum_j|\beta_j| \le \lambda`.

    Projected gradient descent onto the :math:`\ell_1` ball, so the
    constraint holds *exactly* at every iterate rather than
    approximately at the end.
    """
    B = indicator_basis(X, knots, max_order)
    D = B["design"]
    t = [float(v) for v in k.vec(y)]
    n, p = len(D), len(D[0])
    if len(t) != n:
        raise ValueError("tlhal: %d rows but %d outcomes"
                         % (n, len(t)))
    if float(lam) <= 0.0:
        raise ValueError("tlhal: lambda must be positive")
    b = [0.0] * p
    b0 = sum(t) / n if intercept else 0.0
    # step size from the Lipschitz constant of the squared-error
    # gradient, 2 lambda_max(D'D)/n, estimated by power iteration --
    # a fixed step diverges once the basis is large.
    v = [1.0] * p
    lmax = 1.0
    for _ in range(30):
        Dv = [sum(D[i][j] * v[j] for j in range(p)) for i in range(n)]
        w = [sum(D[i][j] * Dv[i] for i in range(n)) for j in range(p)]
        nw = math.sqrt(sum(q * q for q in w))
        if nw <= _EPS:
            break
        v = [q / nw for q in w]
        lmax = nw
    step = min(float(step), 0.9 * n / max(2.0 * lmax, _EPS))
    hist = []
    for _ in range(int(iters)):
        pred = [b0 + sum(D[i][j] * b[j] for j in range(p)
                         if b[j] != 0.0) for i in range(n)]
        res = [pred[i] - t[i] for i in range(n)]
        hist.append(sum(v * v for v in res) / n)
        gr = [2.0 * sum(D[i][j] * res[i] for i in range(n)) / n
              for j in range(p)]
        b = [b[j] - float(step) * gr[j] for j in range(p)]
        if intercept:
            b0 -= float(step) * 2.0 * sum(res) / n
        b = _project_l1(b, float(lam))
    pred = [b0 + sum(D[i][j] * b[j] for j in range(p))
            for i in range(n)]
    return RichResult(payload={
        "estimate": b, "beta": b, "intercept": b0,
        "columns": B["columns"], "n_basis": B["n_basis"],
        "variation_norm": variation_norm(b), "lambda": float(lam),
        "mse": sum((pred[i] - t[i]) ** 2 for i in range(n)) / n,
        "mse_history": hist, "max_order": int(max_order),
        "method": "highly adaptive lasso; van der Laan & Rose (2018) "
                  "Chap. 6",
        "note": "the L1 bound IS the variation norm of the fit, and "
                "the rate beats n^{-1/4} without any smoothness "
                "assumption",
    })


def _project_l1(v, lam):
    """Euclidean projection onto {b : sum |b_j| <= lam}."""
    if sum(abs(x) for x in v) <= lam:
        return v
    u = sorted((abs(x) for x in v), reverse=True)
    css, rho, theta = 0.0, 0, 0.0
    for j in range(len(u)):
        css += u[j]
        if u[j] - (css - lam) / (j + 1) > 0:
            rho = j + 1
            theta = (css - lam) / (j + 1)
    return [math.copysign(max(abs(x) - theta, 0.0), x) for x in v]


def hal_predict(model, X):
    r"""Evaluate the fit at new points."""
    rows = [[float(v) for v in r] for r in k.mat(X)]
    cols, b = model["columns"], model["beta"]
    out = []
    for r in rows:
        s = model["intercept"]
        for j in range(len(cols)):
            if b[j] == 0.0:
                continue
            S, u = cols[j]
            if all(r[S[t]] >= u[t] for t in range(len(S))):
                s += b[j]
        out.append(s)
    return out


def cv_select_lambda(X, y, lambdas, V=5, seed=0, **kw):
    r"""Choose the variation-norm bound by cross-validation."""
    rows = [[float(v) for v in r] for r in k.mat(X)]
    t = [float(v) for v in k.vec(y)]
    n = len(t)
    rng = np.random.default_rng(seed)
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(float(rng.uniform()) * (i + 1)) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    folds = [idx[v::int(V)] for v in range(int(V))]
    risks = {}
    for lam in lambdas:
        tot, m = 0.0, 0
        for f in folds:
            tr = [i for i in range(n) if i not in set(f)]
            fit = hal_fit([rows[i] for i in tr], [t[i] for i in tr],
                          lam=lam, knots=[rows[i] for i in tr], **kw)
            pr = hal_predict(fit, [rows[i] for i in f])
            for a, i in enumerate(f):
                tot += (pr[a] - t[i]) ** 2
                m += 1
        risks[lam] = tot / m
    best = min(sorted(risks), key=lambda v: risks[v])
    return {"lambda": best, "cv_risks": risks,
            "note": "lambda bounds the variation norm, so the tuning "
                    "parameter is interpretable"}


def cheatsheet():
    return ("tlhal: replace SMOOTHNESS with a VARIATION NORM bound. "
            "Any cadlag function of finite variation is a sum over "
            "subsets of integrals against products of indicators, so "
            "fit a linear combination of INDICATOR BASIS functions "
            "under sum|beta| <= lambda -- and for the discrete "
            "approximation that L1 bound IS the variation norm. "
            "Lambda is chosen by cross-validation. The rate beats "
            "n^{-1/4} even in a fully nonparametric model, which is "
            "exactly the threshold double-robust efficiency arguments "
            "require of nuisance estimators.")


# compact alias per ledger/NAMING.md
highlyadaptivelasso = hal_fit
