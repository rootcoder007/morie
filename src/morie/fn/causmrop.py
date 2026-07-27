# morie.fn -- function file (rootcoder007/morie)
"""G-formula (parametric) standardised mean."""

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_robins_g_formula"]


def causal_robins_g_formula(y, A, L, fit_fn=None):
    r"""Point-treatment parametric g-formula (standardisation).

    Under exchangeability given L, the counterfactual mean is

    .. math:: E[Y(a)] = E\big[ E[Y \mid A=a, L] \big],

    the outcome regression evaluated at :math:`A=a` for every unit and
    averaged over the observed L distribution. The default outcome
    model is OLS of y on (A, L, A x L); ``fit_fn`` may replace it.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    A : array-like of {0, 1}, shape (n,)
        Treatment.
    L : array-like, shape (n,) or (n, p)
        Baseline confounders.
    fit_fn : callable, optional
        ``fit_fn(A_col, L, y) -> predict``, where ``predict(a)`` returns
        the fitted E[Y | A=a, L] vector over the sample. Default: OLS
        with treatment-confounder interactions.

    Returns
    -------
    RichResult
        keys: ``EY1``, ``EY0``, ``ate``, ``n``, ``method``.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in
    mortality studies with a sustained exposure period -- application
    to control of the healthy worker survivor effect. *Mathematical
    Modelling*, 7, 1393-1512. (the g-formula)

    Hernan, M. A. & Robins, J. M. (2020). *Causal Inference: What If*.
    Chapman & Hall/CRC. Ch. 13 (standardization and the parametric
    g-formula).
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float).ravel()
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L[:, None]
    n = y.size
    if A.size != n or L.shape[0] != n:
        raise ValueError("y, A, L must share their first dimension.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("A must be binary 0/1.")
    if A.sum() == 0 or A.sum() == n:
        raise ValueError("need both treated and untreated units.")

    if fit_fn is None:

        def fit_fn(a_obs, L, y):
            D = np.column_stack([np.ones(n), a_obs, L, a_obs[:, None] * L])
            b, *_ = np.linalg.lstsq(D, y, rcond=None)

            def predict(a):
                a_col = np.full(n, float(a))
                Da = np.column_stack([np.ones(n), a_col, L, a_col[:, None] * L])
                return Da @ b

            return predict

    predict = fit_fn(A, L, y)
    ey1 = float(np.mean(predict(1)))
    ey0 = float(np.mean(predict(0)))

    return RichResult(
        payload={
            "EY1": ey1,
            "EY0": ey0,
            "ate": ey1 - ey0,
            "n": int(n),
            "method": "G-formula (parametric) standardised mean",
        }
    )


def cheatsheet():
    return "causmrop: E[Y(a)] = mean of fitted E[Y|A=a,L] over observed L"
