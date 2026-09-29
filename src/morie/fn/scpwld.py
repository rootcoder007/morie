# morie.fn -- function file (rootcoder007/morie)
"""Spatial Poisson Wald test on rho."""

from __future__ import annotations

import math

from ._richresult import RichResult


def scpwld(rho, se_rho, rho0=0.0) -> RichResult:
    r"""Wald test of ``rho = rho0`` in a spatial-lag count model.

    ``W = ((rho - rho0) / se(rho))^2`` is asymptotically chi-square with one
    degree of freedom; the p-value is ``P(chi2_1 > W) = erfc(sqrt(W / 2))``.

    Parameters
    ----------
    rho : estimated spatial parameter.
    se_rho : its standard error.
    rho0 : null value (default 0).

    Returns
    -------
    RichResult
        ``statistic``, ``df`` (1), ``p_value`` and ``z`` (the signed root).

    References
    ----------
    Wald, A. (1943). Tests of statistical hypotheses concerning several
    parameters when the number of observations is large. *Transactions of the
    American Mathematical Society* 54, 426-482.

    Examples
    --------
    >>> r = scpwld(0.3, 0.1)
    >>> round(r.statistic, 12), round(r.p_value, 12)
    (9.0, 0.002699796063)
    """
    se = float(se_rho)
    if not se > 0:
        raise ValueError("se_rho must be positive")
    z = (float(rho) - float(rho0)) / se
    w = z * z
    return RichResult(payload={"statistic": w, "df": 1, "p_value": math.erfc(math.sqrt(w / 2.0)), "z": z})


scpwld_fn = scpwld


def cheatsheet() -> str:
    return "scpwld(rho, se_rho, rho0) -> Wald chi-square(1) test of the spatial parameter."
