# morie.fn -- function file (rootcoder007/morie)
r"""The functional linear model :math:`Y = \int \beta(t) X(t)\,dt + \varepsilon`.

The predictor is a curve, not a vector, so the parameter is a *function*
:math:`\beta` and the design has no inverse: the empirical covariance
operator of :math:`X` is compact, its eigenvalues accumulate at zero, and
:math:`\beta` is therefore not identified without regularisation. That is
the whole difficulty of the model and the reason a naive least-squares
solve does not exist.

Cardot, Ferraty and Sarda regularise by TRUNCATION: expand :math:`X` in
its own empirical eigenbasis (functional principal components), keep the
first :math:`k` components, and solve the resulting :math:`k`-dimensional
least-squares problem. With

.. math:: \hat\beta(t) = \sum_{j=1}^{k} \frac{\hat b_j}{\hat\lambda_j}
          \hat\varphi_j(t), \qquad
          \hat b_j = \frac{1}{n}\sum_i \hat\xi_{ij}(Y_i - \bar Y),

where :math:`\hat\lambda_j, \hat\varphi_j` are the eigenvalues and
eigenfunctions of the empirical covariance operator and
:math:`\hat\xi_{ij} = \int (X_i - \bar X)\hat\varphi_j` the scores, the
estimator is consistent provided :math:`k = k(n)` grows slowly enough.

**The truncation level is the estimator, not a detail.** Dividing by
:math:`\hat\lambda_j` amplifies exactly the directions the data constrain
least, so every additional component buys fit and costs stability. The
returned ``eigenvalues`` and ``explained`` make that trade visible rather
than leaving it implicit in a default.

Both routes the literature uses are available and neither is hidden:
passing an integer selects the truncation level directly; passing a basis
matrix (one column per basis function, evaluated on the same grid)
projects onto that basis first, which is the spline route of the same
paper. The default is the integer route because the eigenbasis is the one
the consistency result is stated in.

References
----------
Cardot, H., Ferraty, F. and Sarda, P. (1999) "Functional linear model",
*Statistics & Probability Letters* **45**(1), 11-22,
doi:10.1016/S0167-7152(99)00036-X. The model, the spline and functional
principal component estimators, and the consistency of the truncated
estimator.

Ramsay, J. O. and Silverman, B. W. (2005) *Functional Data Analysis*,
2nd ed., Springer, Ch. 15 (the functional linear model with a functional
covariate and a scalar response; the ill-posedness of the normal
equations).

Hall, P. and Horowitz, J. L. (2007) "Methodology and convergence rates
for functional linear regression", *Annals of Statistics* **35**(1),
70-91, doi:10.1214/009053606000000957. Minimax rates, and why the
truncation level cannot be chosen independently of the eigenvalue decay.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["functional_regression"]

_EPS = 1e-12


def _grid_weights(n_t):
    """Trapezoid weights on a unit grid: the discrete stand-in for dt."""
    if n_t < 2:
        return [1.0]
    h = 1.0 / (n_t - 1)
    w = [h] * n_t
    w[0] = 0.5 * h
    w[-1] = 0.5 * h
    return w


def functional_regression(X, Y, basis=None):
    r"""Fit the functional linear model by principal-component truncation.

    Parameters
    ----------
    X : (n, T) array-like
        One curve per row, all on a common grid.
    Y : (n,) array-like
        Scalar response.
    basis : int or (T, p) array-like, optional
        An integer truncates at that many functional principal
        components. A matrix projects onto that basis first (one column
        per basis function on the same grid) and truncates at its rank.
        Defaults to the number of components explaining 99% of the
        curve-to-curve variance, reported as ``k``.

    Returns
    -------
    RichResult with ``beta`` on the grid, ``fitted``, ``residuals``,
    ``k``, ``eigenvalues``, ``explained``, ``scores``, ``r_squared``.
    """
    Xm = [[float(v) for v in row] for row in k.mat(X)]
    y = [float(v) for v in k.vec(Y)]
    n = len(Xm)
    if n == 0:
        raise ValueError("rgs: no curves")
    if len(y) != n:
        raise ValueError("rgs: %d curves but %d responses" % (n, len(y)))
    T = len(Xm[0])
    if any(len(r) != T for r in Xm):
        raise ValueError("rgs: every curve must lie on the same grid")
    if n < 2:
        raise ValueError("rgs: the covariance operator needs at least two "
                         "curves")
    w = _grid_weights(T)

    # optional basis projection -- the spline route of the same paper
    B = None
    kk = None
    if basis is not None and not isinstance(basis, (int, float, bool)):
        B = [[float(v) for v in row] for row in k.mat(basis)]
        if len(B) != T:
            raise ValueError("rgs: the basis has %d rows for a grid of %d"
                             % (len(B), T))
    elif basis is not None:
        kk = int(basis)
        if kk < 1:
            raise ValueError("rgs: the truncation level must be at least 1")

    xbar = [sum(Xm[i][t] for i in range(n)) / n for t in range(T)]
    Xc = [[Xm[i][t] - xbar[t] for t in range(T)] for i in range(n)]

    # Empirical covariance operator, discretised with the grid weights so
    # the eigenproblem is the one for the integral operator rather than
    # for the raw matrix.
    C = [[sum(Xc[i][s] * Xc[i][t] for i in range(n)) / n
          for t in range(T)] for s in range(T)]
    rw = [math.sqrt(v) for v in w]
    Cw = [[rw[s] * C[s][t] * rw[t] for t in range(T)] for s in range(T)]
    lam, U = k.jacobi(Cw)                     # eigenvectors are COLUMNS
    order = sorted(range(T), key=lambda j: -lam[j])
    lam = [max(lam[j], 0.0) for j in order]
    phi = [[U[s][j] / (rw[s] if rw[s] > _EPS else 1.0) for s in range(T)]
           for j in order]                    # phi[j] is an eigenfunction
    # An eigenvector is defined only up to sign, so the SCORES would be
    # reported with an arbitrary sign unless it is pinned. Make the entry of
    # largest magnitude positive; beta and fitted are unaffected either way
    # because b_j flips with phi_j.
    for j in range(len(phi)):
        top = max(range(T), key=lambda t: abs(phi[j][t]))
        if phi[j][top] < 0.0:
            phi[j] = [-v for v in phi[j]]

    total = sum(lam)
    if total <= _EPS:
        raise ValueError("rgs: the curves carry no variation to regress on")
    explained = [v / total for v in lam]

    if B is not None:
        kk = min(len(B[0]), T)
    if kk is None:
        run, kk = 0.0, T
        for j in range(T):
            run += explained[j]
            if run >= 0.99:
                kk = j + 1
                break
    kk = max(1, min(kk, T, n - 1))
    # a component with no variance cannot be divided by
    kk = max(1, len([j for j in range(kk) if lam[j] > _EPS * total]))

    scores = [[sum(Xc[i][t] * phi[j][t] * w[t] for t in range(T))
               for j in range(kk)] for i in range(n)]
    ybar = sum(y) / n
    b = []
    for j in range(kk):
        cov = sum(scores[i][j] * (y[i] - ybar) for i in range(n)) / n
        b.append(cov / lam[j])
    beta = [sum(b[j] * phi[j][t] for j in range(kk)) for t in range(T)]

    fitted = [ybar + sum(b[j] * scores[i][j] for j in range(kk))
              for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]
    sst = sum((v - ybar) ** 2 for v in y)
    sse = sum(v * v for v in resid)
    r2 = 1.0 - sse / sst if sst > _EPS else 0.0

    return RichResult(payload={
        "estimate": beta,
        "beta": beta,
        "fitted": fitted,
        "residuals": resid,
        "k": int(kk),
        "eigenvalues": lam[:kk],
        "explained": explained[:kk],
        "scores": scores,
        "mean_curve": xbar,
        "r_squared": r2,
        "n": n,
        "n_grid": T,
        "method": "functional linear model by principal-component "
                  "truncation (Cardot, Ferraty & Sarda 1999)",
        "note": "beta is divided by the eigenvalues, so each extra "
                "component amplifies a direction the data constrain "
                "less -- k is the estimator, not a detail",
    })


def cheatsheet():
    return ("rgs: functional_regression(X, Y, basis) -> the functional "
            "linear model Y = int beta(t) X(t) dt by FPC truncation "
            "(Cardot, Ferraty & Sarda 1999, Stat. Probab. Lett. 45(1), "
            "11-22)")
