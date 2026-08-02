# morie.fn -- function file (rootcoder007/morie)
"""Partly linear logistic regression."""

from . import _array_core as np

from scipy import optimize

from ._richresult import RichResult

__all__ = ["kosorok_ch1_partly_linear_logistic"]


def kosorok_ch1_partly_linear_logistic(Y, Z, U, beta=None, eta=None, df=5):
    r"""Partly linear logistic model (Kosorok Ch. 1):

    .. math:: E[Y \mid Z, U] = \nu(\beta' Z + \eta(U)), \qquad
              \nu(t) = \frac{1}{1 + e^{-t}}.

    The parametric part beta is what the semiparametric theory targets;
    eta is an infinite-dimensional nuisance, approximated here by a
    natural cubic spline basis in U with ``df`` degrees of freedom.
    The whole point of the example is that beta remains root-n
    estimable even though eta converges more slowly, so the fit
    returns both and does not pretend eta is a parameter of interest.

    Parameters
    ----------
    Y : array-like of {0, 1}, shape (n,)
        Binary response.
    Z : array-like, shape (n,) or (n, p)
        Parametric covariates.
    U : array-like, shape (n,)
        The covariate entering nonparametrically.
    beta, eta : ignored
        Interface compatibility; both are estimated.
    df : int, default 5
        Spline degrees of freedom for eta.

    Returns
    -------
    RichResult
        keys: ``beta``, ``eta_coef``, ``eta_fitted``, ``loglik``,
        ``converged``, ``n``, ``df``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 1 (partly linear logistic regression).
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    U = np.asarray(U, dtype=float).ravel()
    n, p = Z.shape
    if Y.size != n or U.size != n:
        raise ValueError("Y, Z and U must have the same number of rows.")
    if not np.all(np.isin(Y, (0.0, 1.0))):
        raise ValueError("Y must be binary 0/1.")
    df = int(df)
    if df < 2 or df >= n:
        raise ValueError(f"df must lie in 2..{n - 1}, got {df}.")

    # truncated power basis for eta, knots at interior quantiles
    knots = np.quantile(U, np.linspace(0, 1, df)[1:-1]) if df > 2 else np.array([])
    B = [np.ones(n), U]
    for k in knots:
        B.append(np.maximum(U - k, 0.0) ** 3)
    B = np.column_stack(B)
    X = np.column_stack([Z, B])

    def neg(par):
        lin = X @ par
        lin = np.clip(lin, -30, 30)
        return -float(np.sum(Y * lin - np.log1p(np.exp(lin))))

    res = optimize.minimize(neg, np.zeros(X.shape[1]), method="BFGS")
    par = res.x
    lin = np.clip(X @ par, -30, 30)
    return RichResult(
        payload={"beta": par[:p], "eta_coef": par[p:],
                 "eta_fitted": B @ par[p:], "loglik": float(-res.fun),
                 "converged": bool(res.success), "n": int(n), "df": df,
                 "method": "Partly linear logistic; eta by spline basis (Kosorok Ch. 1)"}
    )


def cheatsheet():
    return "ksr024: beta root-n estimable despite the slower nuisance eta"
