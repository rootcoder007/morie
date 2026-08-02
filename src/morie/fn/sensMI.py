# morie.fn -- function file (rootcoder007/morie)
"""Sensitivity of the ACME to an unobserved mediator-outcome confounder."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sensitivity_mediation_imbens"]


def _ols(X, y):
    """Least squares with an intercept prepended; returns (beta, resid)."""
    D = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return beta, y - D @ beta


def sensitivity_mediation_imbens(Y, X, C, r2_grid=None):
    r"""Sensitivity analysis for causal mediation under the LSEM.

    Fits the Baron-Kenny system, equations (11)-(13) of Imai, Keele &
    Yamamoto (2010):

    .. math::

        Y_i &= \alpha_1 + \beta_1 T_i + \varepsilon_{i1} \\
        M_i &= \alpha_2 + \beta_2 T_i + \varepsilon_{i2} \\
        Y_i &= \alpha_3 + \beta_3 T_i + \gamma M_i + \varepsilon_{i3}

    Sequential ignorability makes the ACME equal to
    :math:`\beta_2\gamma` (their Theorem 2). That assumption fails if an
    unobserved pre-treatment variable confounds the mediator-outcome
    relation. Their Theorem 4 identifies the ACME for any *given* value
    of :math:`\rho = \mathrm{Corr}(\varepsilon_{i2},\varepsilon_{i3})`:

    .. math::

        \bar\delta(\rho) = \frac{\beta_2\sigma_1}{\sigma_2}
            \left[\tilde\rho - \rho\sqrt{\frac{1-\tilde\rho^2}{1-\rho^2}}\right]

    with :math:`\sigma_j^2 = \mathrm{Var}(\varepsilon_{ij})` and
    :math:`\tilde\rho = \mathrm{Corr}(\varepsilon_{i1},\varepsilon_{i2})`.
    The ACME is zero exactly at :math:`\rho = \tilde\rho`, which makes
    :math:`\tilde\rho` the breakdown point of the finding.

    Writing the confounder explicitly as
    :math:`\varepsilon_{ij} = \lambda_j U_i + \varepsilon'_{ij}` gives the
    partial-R^2 reading of the same parameter -- the share of otherwise
    unexplained variance that U accounts for. Imai et al. attribute this
    parameterisation to Imbens (2003):

    .. math::

        R^{2*}_M = 1 - \frac{\mathrm{Var}(\varepsilon'_{i2})}{\mathrm{Var}(\varepsilon_{i2})},
        \qquad
        R^{2*}_Y = 1 - \frac{\mathrm{Var}(\varepsilon'_{i3})}{\mathrm{Var}(\varepsilon_{i3})},
        \qquad
        \rho^2 = R^{2*}_M R^{2*}_Y

    Parameters
    ----------
    Y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n,)
        Treatment. One-dimensional -- each equation above carries a
        single treatment coefficient.
    C : array-like, shape (n,)
        Mediator.
    r2_grid : array-like, optional
        Values of the *product* :math:`R^{2*}_M R^{2*}_Y`, each in
        [0, 1). Each entry gives :math:`|\rho| = \sqrt{r}` by the
        identity above. The ACME is reported at both signs, because the
        sign of :math:`\rho` follows :math:`\mathrm{sgn}(\lambda_2\lambda_3)`,
        which the data cannot reveal. Defaults to 10 points on [0, 0.81].

    Returns
    -------
    RichResult
        keys: ``estimate`` (the ACME under sequential ignorability,
        :math:`\beta_2\gamma`), ``rho_breakdown`` (:math:`\tilde\rho`),
        ``r2_grid``, ``rho_grid``, ``acme_positive``, ``acme_negative``,
        ``beta2``, ``gamma``, ``sigma1``, ``sigma2``, ``n``, ``method``.

    References
    ----------
    Imai, K., Keele, L. & Yamamoto, T. (2010). Identification, inference
    and sensitivity analysis for causal mediation effects. *Statistical
    Science*, 25(1), 51-71.

    Imbens, G. W. (2003). Sensitivity to exogeneity assumptions in
    program evaluation. *American Economic Review*, 93(2), 126-132.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    T = np.asarray(X, dtype=float)
    M = np.asarray(C, dtype=float).ravel()
    if T.ndim != 1:
        raise ValueError(f"Treatment X must be one-dimensional, got shape {T.shape}.")
    n = Y.size
    if not (T.size == n and M.size == n):
        raise ValueError(f"Y, X and C must be the same length; got {n}, {T.size}, {M.size}.")
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}.")

    _, e1 = _ols(T.reshape(-1, 1), Y)  # eq (11)
    b2, e2 = _ols(T.reshape(-1, 1), M)  # eq (12)
    b3, _ = _ols(np.column_stack([T, M]), Y)  # eq (13)
    beta2 = float(b2[1])
    gamma = float(b3[2])

    sigma1 = float(np.std(e1, ddof=0))
    sigma2 = float(np.std(e2, ddof=0))
    if sigma2 <= 0:
        raise ValueError("Mediator is perfectly explained by the treatment; rho is undefined.")
    rho_tilde = float(np.corrcoef(e1, e2)[0, 1])

    if r2_grid is None:
        r2_grid = np.linspace(0.0, 0.81, 10)
    r2 = np.asarray(r2_grid, dtype=float).ravel()
    if np.any(r2 < 0) or np.any(r2 >= 1):
        raise ValueError("r2_grid holds a product of two R^2 values; each entry must lie in [0, 1).")

    rho = np.sqrt(r2)
    scale = beta2 * sigma1 / sigma2

    def acme(rv):
        return scale * (rho_tilde - rv * np.sqrt((1 - rho_tilde**2) / (1 - rv**2)))

    return RichResult(
        title="Mediation sensitivity to an unobserved confounder",
        payload={
            "estimate": beta2 * gamma,
            "rho_breakdown": rho_tilde,
            "r2_grid": r2,
            "rho_grid": rho,
            "acme_positive": acme(rho),
            "acme_negative": acme(-rho),
            "beta2": beta2,
            "gamma": gamma,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "n": int(n),
            "method": "LSEM mediation sensitivity (Imai-Keele-Yamamoto Thm 4)",
        },
    )


def cheatsheet():
    return "sensMI: mediation ACME sensitivity to an unobserved confounder"
