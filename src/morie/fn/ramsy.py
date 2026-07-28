# morie.fn -- function file (rootcoder007/morie)
"""Ramsey RESET test for functional form."""

# NOTE: the computation lives in morie.fn.rsetf.reset_core. This
# module previously carried its own copy, which raised UNSCALED
# fitted values to powers; cubing them makes the condition number
# of the augmented design grow with the sixth power of the response
# scale, and the same defect was present in morie.fn.reset,
# morie.fn.rmsyt, morie.fn.ramsy and the R implementation. One core,
# one fix.


from __future__ import annotations

import numpy as np

from ._containers import TestResult
from .rsetf import reset_core


def ramsey_reset_test(
    X,
    y,
    *,
    power: int = 3,
) -> TestResult:
    """Ramsey RESET test for omitted nonlinearities.

    Augments the OLS model with powers of fitted values
    (y-hat^2, ..., y-hat^power) and tests joint significance.

    Parameters
    ----------
    X : array-like, shape (n, p) or (n,)
        Design matrix.
    y : array-like, shape (n,)
        Response.
    power : int
        Maximum power of fitted values (default 3).

    Returns
    -------
    TestResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    X_int = np.column_stack([np.ones(n), X])
    powers = list(range(2, int(power) + 1))
    f, pv, ssr_r, ssr_u, df1, df2, n, _ = reset_core(y, X_int, powers)
    return TestResult(
        test_name="Ramsey RESET",
        statistic=float(f),
        p_value=float(pv),
        df=float(df1),
        method=f"RESET with powers 2..{int(power)}",
        n=n,
        extra={"df_num": df1, "df_den": df2, "powers": powers,
               "ssr_restricted": ssr_r, "ssr_unrestricted": ssr_u},
    )


def cheatsheet() -> str:
    return "ramsey_reset_test(X, y) -> Ramsey RESET test."
