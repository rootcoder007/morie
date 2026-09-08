# morie.fn -- function file (rootcoder007/morie)
"""Imai-Keele sensitivity of the ACME to unmeasured confounding."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["imai_sensitivity_rho"]


def _lsem_fit(x, m, y, c=None):
    r"""Fit the three LSEM equations and return the pieces Theorem 2 needs.

    .. math::
        Y &= \alpha_1 + \beta_1 T + \varepsilon_1 \\
        M &= \alpha_2 + \beta_2 T + \varepsilon_2 \\
        Y &= \alpha_3 + \beta_3 T + \gamma M + \varepsilon_3
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if not (m.size == n and y.size == n):
        raise ValueError("x, m, y must have equal length.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but x has {n}.")
    if n < C.shape[1] + 5:
        raise ValueError("too few observations for the three LSEM regressions.")

    def ols(D, t):
        b, *_ = np.linalg.lstsq(D, t, rcond=None)
        return b, t - D @ b

    one = np.ones(n)
    b1, e1 = ols(np.column_stack([one, x, C]), y)
    b2, e2 = ols(np.column_stack([one, x, C]), m)
    b3, e3 = ols(np.column_stack([one, x, m, C]), y)
    return {
        "beta1": float(b1[1]),
        "beta2": float(b2[1]),
        "beta3": float(b3[1]),
        "gamma": float(b3[2]),
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "sigma1": float(e1.std(ddof=1)),
        "sigma2": float(e2.std(ddof=1)),
        "rho_tilde": float(np.corrcoef(e1, e2)[0, 1]),
        "n": n,
    }


def imai_sensitivity_rho(x, m, y, rho_grid=None, c=None):
    r"""ACME as a function of the sensitivity parameter rho.

    Imai, Keele and Tingley's Theorem 2 (extending Imai-Keele-Yamamoto
    2010): letting :math:`\rho = \mathrm{Corr}(\varepsilon_2,
    \varepsilon_3)` be the correlation between the mediator- and
    outcome-model errors -- zero exactly under sequential ignorability
    -- the average causal mediation effect is identified as

    .. math:: \bar\delta(t) = \frac{\beta_2 \sigma_1}{\sigma_2}
              \left\{ \tilde\rho
              - \rho \sqrt{\frac{1 - \tilde\rho^2}{1 - \rho^2}} \right\},

    with :math:`\sigma_j^2 = \mathrm{Var}(\varepsilon_j)` and
    :math:`\tilde\rho = \mathrm{Corr}(\varepsilon_1, \varepsilon_2)`
    estimated from the residuals. At :math:`\rho = 0` this reduces to
    the product-of-coefficients estimate; the curve shows how large an
    unmeasured mediator-outcome confounder would have to be to change
    the conclusion.

    Parameters
    ----------
    x, m, y : array-like, shape (n,)
        Treatment, mediator, outcome.
    rho_grid : array-like in (-1, 1), optional
        Where to evaluate. Default: 41 points on [-0.9, 0.9].
    c : array-like, optional
        Baseline covariates entering all three equations.

    Returns
    -------
    RichResult
        keys: ``rho`` (grid), ``acme`` (matching array), ``acme_0``
        (the rho = 0 value), ``rho_critical`` (where the ACME crosses
        zero, equal to ``rho_tilde``), ``rho_tilde``, ``beta2``,
        ``gamma``, ``sigma1``, ``sigma2``, ``n``, ``method``.

    References
    ----------
    Imai, K., Keele, L. & Tingley, D. (2010). A general approach to
    causal mediation analysis. *Psychological Methods*, 15(4),
    309-334. Theorem 2, p. 316.

    Imai, K., Keele, L. & Yamamoto, T. (2010). Identification,
    inference and sensitivity analysis for causal mediation effects.
    *Statistical Science*, 25(1), 51-71.
    """
    f = _lsem_fit(x, m, y, c=c)
    grid = np.linspace(-0.9, 0.9, 41) if rho_grid is None else np.asarray(rho_grid, dtype=float).ravel()
    if np.any(np.abs(grid) >= 1):
        raise ValueError("rho must lie strictly in (-1, 1).")

    rt = f["rho_tilde"]
    scale = f["beta2"] * f["sigma1"] / f["sigma2"]
    acme = scale * (rt - grid * np.sqrt((1 - rt**2) / (1 - grid**2)))

    return RichResult(
        payload={
            "rho": grid,
            "acme": acme,
            "acme_0": float(scale * rt),
            "rho_critical": rt,  # delta(t) = 0 exactly at rho = rho_tilde
            "rho_tilde": rt,
            "beta2": f["beta2"],
            "gamma": f["gamma"],
            "sigma1": f["sigma1"],
            "sigma2": f["sigma2"],
            "n": int(f["n"]),
            "method": "Imai-Keele-Tingley sensitivity: ACME(rho), Theorem 2",
        }
    )


def cheatsheet():
    return "sensIM: ACME(rho) = (b2 s1/s2){rt - rho sqrt((1-rt^2)/(1-rho^2))} (IKT Thm 2)"
