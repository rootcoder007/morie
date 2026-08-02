# morie.fn -- function file (rootcoder007/morie)
"""Rho critical value where the mediation effect goes to zero."""

from . import _array_core as np

from ._richresult import RichResult
from .sensIM import _lsem_fit

__all__ = ["rho_critical_mediation"]


def rho_critical_mediation(x, m, y, c=None):
    r"""The sensitivity parameter at which the ACME vanishes.

    Setting Imai-Keele-Tingley's Theorem 2 expression to zero,

    .. math:: \tilde\rho = \rho \sqrt{\frac{1-\tilde\rho^2}{1-\rho^2}}
              \iff \rho^2 = \tilde\rho^2,

    so the ACME crosses zero exactly at :math:`\rho = \tilde\rho =
    \mathrm{Corr}(\varepsilon_1, \varepsilon_2)`, estimated by the
    sample correlation of the total-effect and mediator residuals
    (the paper's footnote 6). :math:`|\rho^*|` near zero means a tiny
    unmeasured mediator-outcome confounder would overturn the finding.

    Parameters
    ----------
    x, m, y : array-like, shape (n,)
        Treatment, mediator, outcome.
    c : array-like, optional
        Baseline covariates.

    Returns
    -------
    RichResult
        keys: ``rho_critical``, ``abs_rho_critical`` (the robustness
        summary), ``acme_0``, ``n``, ``method``.

    References
    ----------
    Imai, K., Keele, L. & Tingley, D. (2010). A general approach to
    causal mediation analysis. *Psychological Methods*, 15(4),
    309-334. Theorem 2 and footnote 6, p. 316.
    """
    f = _lsem_fit(x, m, y, c=c)
    rt = f["rho_tilde"]
    return RichResult(
        payload={
            "rho_critical": rt,
            "abs_rho_critical": abs(rt),
            "acme_0": float(f["beta2"] * f["sigma1"] / f["sigma2"] * rt),
            "n": int(f["n"]),
            "method": "rho at which the ACME crosses zero (= Corr(e1, e2))",
        }
    )


def cheatsheet():
    return "rhomed: rho* = Corr(e1, e2); |rho*| small = fragile mediation claim"
