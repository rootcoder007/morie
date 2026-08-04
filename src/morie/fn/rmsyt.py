# morie.fn -- function file (rootcoder007/morie)
"""Ramsey RESET specification test."""

# NOTE: the computation lives in morie.fn.rsetf.reset_core. This
# module previously carried its own copy, which raised UNSCALED
# fitted values to powers; cubing them makes the condition number
# of the augmented design grow with the sixth power of the response
# scale, and the same defect was present in morie.fn.reset,
# morie.fn.rmsyt, morie.fn.ramsy and the R implementation. One core,
# one fix.


from __future__ import annotations

from . import _array_core as np

from ._containers import TestResult
from .rsetf import reset_core


def ramsey_reset(y: np.ndarray, X: np.ndarray, cdf=None, *, powers: list[int] | None = None) -> TestResult:
    """Ramsey RESET test for functional form misspecification.

    Parameters
    ----------
    y : (n,)
    X : (n, p)
    powers : list of int
        Powers of fitted values to include (default [2, 3]).

    Returns
    -------
    TestResult
    """
    if powers is None:
        powers = [2, 3]
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    X_int = np.column_stack([np.ones(n), X])
    f, pv, ssr_r, ssr_u, df1, df2, n, _ = reset_core(y, X_int, powers)
    return TestResult(
        test_name="RESET",
        statistic=float(f),
        p_value=float(pv),
        df=float(df1),
        method="Ramsey RESET",
        n=n,
        extra={"df_num": df1, "df_den": df2, "powers": list(powers),
               "ssr_restricted": ssr_r, "ssr_unrestricted": ssr_u},
    )


def cheatsheet() -> str:
    return "ramsey_reset({}) -> Ramsey RESET specification test."


# compact alias per ledger/NAMING.md
ramseyreset = ramsey_reset
