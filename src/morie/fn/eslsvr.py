"""Support-vector regression with the epsilon-insensitive loss (ESL sec 12.3.6)."""

from . import _array_core as np
from ._richresult import RichResult
from ._svm import kernel_matrix, smo

__all__ = ["esl_svr"]


def esl_svr(X, y, C=1.0, epsilon=0.1, kernel="rbf", gamma=None, degree=3, coef0=1.0, newdata=None, tol=1e-3):
    r"""Minimise :math:`C\sum_i V_\epsilon(y_i - f(x_i)) + \tfrac12\|\beta\|^2` with :math:`V_\epsilon(r) = (|r| - \epsilon)_+`.

    ESL eqs 12.35-12.39: the solution is :math:`\hat f(x) = \sum_i(\hat\alpha^*_i -
    \hat\alpha_i)K(x, x_i) + \beta_0` (the sign of each weight depends only on the
    naming of the two multipliers) with :math:`0 \le \alpha_i, \alpha^*_i \le C` from
    the dual, solved here as a 2N-variable box-constrained quadratic
    programme by the same SMO as the classifiers (the libsvm formulation of
    epsilon-regression), so it matches ``e1071::svm(type = "eps-regression")``.
    Points strictly inside the epsilon-tube have zero dual weight.

    Parameters
    ----------
    X : n x p nested sequence
    y : sequence of n floats
    C : float
        Cost, > 0.
    epsilon : float
        Tube half-width, >= 0.
    kernel : {"linear", "rbf", "poly", "sigmoid"}
    gamma, degree, coef0
        Kernel parameters (gamma defaults to 1/p).
    newdata : m x p nested sequence, optional
    tol : float
        SMO stopping tolerance on the KKT violation.

    Returns
    -------
    RichResult
        ``coef`` (signed dual weight per training point, the coefficient of
        K(x, x_i) in f), ``b``, ``fitted``,
        ``prediction`` (for newdata), ``n_support``, ``converged``.

    References
    ----------
    Hastie, Tibshirani & Friedman (2009), sec. 12.3.6; Smola, A. & Scholkopf,
    B. (2004). A tutorial on support vector regression. Statistics and
    Computing 14, 199-222.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    yv = [float(v) for v in np.asarray(y, dtype=float).ravel()]
    n = len(yv)
    if X.shape[0] != n or C <= 0 or epsilon < 0:
        raise ValueError("need matching X and y, C > 0 and epsilon >= 0")
    K = kernel_matrix(X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    Kl = [[float(K[i, j]) for j in range(n)] for i in range(n)]
    K2 = [[Kl[i % n][j % n] for j in range(2 * n)] for i in range(2 * n)]
    z = [1.0] * n + [-1.0] * n
    pvec = [epsilon - v for v in yv] + [epsilon + v for v in yv]
    a, b, it, conv = smo(np.asarray(K2), np.asarray(z), C=C, tol=tol, p=pvec)
    a = [float(v) for v in a]
    coef = [a[i] - a[n + i] for i in range(n)]
    fitted = [sum(coef[j] * Kl[i][j] for j in range(n)) + b for i in range(n)]
    pred = None
    if newdata is not None:
        Kn = kernel_matrix(
            np.atleast_2d(np.asarray(newdata, dtype=float)), X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0
        )
        pred = [sum(coef[j] * float(Kn[i, j]) for j in range(n)) + b for i in range(Kn.shape[0])]
    return RichResult(
        title="Support-vector regression",
        summary_lines=[("n_support", sum(abs(c) > 1e-12 for c in coef))],
        payload={
            "coef": coef,
            "b": b,
            "fitted": fitted,
            "prediction": pred,
            "n_support": sum(abs(c) > 1e-12 for c in coef),
            "converged": conv,
            "n_iter": it,
        },
    )


def cheatsheet():
    return "eslsvr: eps-insensitive SVR via the 2N-variable dual and SMO (libsvm form); f = sum (a* - a) K + b"
