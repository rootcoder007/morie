# morie.fn -- function file (rootcoder007/morie)
"""Bias correction for exposure misclassification."""

import math

from ._richresult import RichResult

__all__ = ["exposure_misclass_bias"]


def exposure_misclass_bias(A_obs, Se, Sp, N=None):
    """
    Bias correction for exposure misclassification

    Formula: under non-differential or group-specific misclassification
    with sensitivity Se and specificity Sp, the expected observed count
    of exposed subjects in a group of size N whose true exposed count is
    A is

        E[A_obs] = Se A + (1 - Sp) (N - A)

    so the matrix-corrected true count is obtained by inverting it,

        A = (A_obs - (1 - Sp) N) / (Se + Sp - 1)

    which is the "divide by (Se + Sp - 1)" rule of the docstring written
    out with its offset term.  The correction is undefined when
    Se + Sp = 1 (the classification carries no information) and is
    inadmissible when Se + Sp < 1.

    With exactly two groups -- conventionally cases first, then controls
    -- the corrected and uncorrected odds ratios are both reported.

    Parameters
    ----------
    A_obs : array-like
        Observed exposed count in each group.
    Se : float or array-like
        Sensitivity, scalar or one per group, in (0, 1].
    Sp : float or array-like
        Specificity, scalar or one per group, in (0, 1].
    N : array-like, optional
        Group totals.  If omitted, ``A_obs`` must be length 2 and is read
        as (exposed, unexposed) of a single group.

    Returns
    -------
    result : dict
        Keys: estimate (first corrected count), a_true, a_obs, totals,
        prevalence, or_obs, or_true, sensitivity, specificity, n, method.

    References
    ----------
    Greenland (1988), Statistics in Medicine 7(7):745-757,
    doi:10.1002/sim.4780070704.
    """
    a = [float(v) for v in ([A_obs] if isinstance(A_obs, (int, float)) else A_obs)]
    if N is None:
        if len(a) != 2:
            raise ValueError("N is required unless A_obs is (exposed, unexposed)")
        tot = [a[0] + a[1]]
        a = [a[0]]
    else:
        tot = [float(v) for v in ([N] if isinstance(N, (int, float)) else N)]
        if len(tot) != len(a):
            raise ValueError("A_obs and N must have the same length")
    g = len(a)
    if g == 0:
        raise ValueError("empty input: A_obs has no groups")
    se = [float(Se)] * g if isinstance(Se, (int, float)) else [float(v) for v in Se]
    sp = [float(Sp)] * g if isinstance(Sp, (int, float)) else [float(v) for v in Sp]
    if len(se) != g or len(sp) != g:
        raise ValueError("Se and Sp must be scalars or one value per group")
    for i in range(g):
        if not (0.0 < se[i] <= 1.0) or not (0.0 < sp[i] <= 1.0):
            raise ValueError("Se and Sp must lie in (0, 1]")
        if se[i] + sp[i] <= 1.0:
            raise ValueError("Se + Sp must exceed 1 for the correction to invert")
        if tot[i] <= 0.0:
            raise ValueError("group totals must be positive")
        if a[i] < 0.0 or a[i] > tot[i]:
            raise ValueError("A_obs must lie between 0 and the group total")
    at = [(a[i] - (1.0 - sp[i]) * tot[i]) / (se[i] + sp[i] - 1.0) for i in range(g)]
    prev = [at[i] / tot[i] for i in range(g)]

    def _or(x):
        n0 = tot[0] - x[0]
        n1 = tot[1] - x[1]
        if x[1] <= 0.0 or n0 <= 0.0 or n1 <= 0.0 or x[0] <= 0.0:
            return float("nan")
        return (x[0] * n1) / (x[1] * n0)

    oro = _or(a) if g == 2 else float("nan")
    ort = _or(at) if g == 2 else float("nan")
    return RichResult(payload={
        "estimate": at[0],
        "a_true": at,
        "a_obs": a,
        "totals": tot,
        "prevalence": prev,
        "or_obs": oro,
        "or_true": ort,
        "sensitivity": se,
        "specificity": sp,
        "n": g,
        "method": "Bias correction for exposure misclassification",
    })


def cheatsheet():
    return "epbias: Bias correction for exposure misclassification"


# compact alias per ledger/NAMING.md
exposuremisclassbias = exposure_misclass_bias
