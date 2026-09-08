# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic null variance of a linear rank statistic."""

from . import _array_core as np
from ._sci_core import integrate

from ._richresult import RichResult

__all__ = ["gibbons_cs_null_var"]


def gibbons_cs_null_var(J, lam):
    r"""Corollary 7.3.1 (Chernoff-Savage form): for a two-sample
    linear rank statistic with score-generating function J and
    limiting sample fraction :math:`\lambda_N = m/N \to \lambda`,

    .. math:: N \lambda_N \sigma_N^2 \;\to\;
              (1 - \lambda)\Big[\int_0^1 J^2(u)\,du -
              \Big(\int_0^1 J(u)\,du\Big)^2\Big].

    The bracket is just the variance of J(U) for uniform U; the
    (1 - lambda) factor is what the two-sample split contributes.

    Parameters
    ----------
    J : callable
        Score-generating function on (0, 1).
    lam : float in (0, 1)
        Limiting fraction m/N.

    Returns
    -------
    RichResult
        keys: ``limit`` (the RHS), ``var_J`` (Var J(U)), ``mean_J``,
        ``lam``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Corollary 7.3.1.

    Chernoff, H. & Savage, I. R. (1958). Asymptotic normality and
    efficiency of certain nonparametric test statistics. *The Annals
    of Mathematical Statistics*, 29(4), 972-994.
    """
    lam = float(lam)
    if not 0 < lam < 1:
        raise ValueError(f"lam must lie in (0, 1), got {lam}.")
    mJ, _ = integrate.quad(J, 0, 1, limit=200)
    mJ2, _ = integrate.quad(lambda u: J(u) ** 2, 0, 1, limit=200)
    varJ = mJ2 - mJ**2
    if varJ < 0:
        varJ = 0.0
    return RichResult(
        payload={
            "limit": float((1 - lam) * varJ), "var_J": float(varJ),
            "mean_J": float(mJ), "lam": lam,
            "method": "N lam sigma^2 -> (1-lam) Var J(U) (Corollary 7.3.1)",
        }
    )


def cheatsheet():
    return "gb7381: limit = (1-lam)[int J^2 - (int J)^2]; Var of J(U)"
