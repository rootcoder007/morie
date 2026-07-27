# morie.fn -- function file (rootcoder007/morie)
"""Generalized Synthetic Control with interactive fixed effects."""

import numpy as np

from ._richresult import RichResult

__all__ = ["generalized_synthetic_control"]


def generalized_synthetic_control(y_treated, y_controls, treat_time, r=2):
    r"""Xu's generalised synthetic control via a latent factor model.

    Interactive fixed effects: :math:`Y_{it}(0) = \lambda_i' f_t +
    \varepsilon_{it}`. The factors :math:`f_t` are estimated from the
    (demeaned) control panel by principal components; the treated
    unit's loadings :math:`\lambda_1` come from regressing its
    pre-treatment outcomes on the pre-treatment factors; the
    counterfactual :math:`\hat Y_{1t}(0)` is imputed for the post
    period and ATT is the mean post-period gap. Unlike simplex SCM
    this extrapolates freely, so it also works when the treated unit
    lies outside the donor convex hull.

    Parameters
    ----------
    y_treated : array-like, shape (T,)
        Treated unit's outcome series.
    y_controls : array-like, shape (T, J)
        Donor (never-treated) outcomes.
    treat_time : int
        First post-treatment index.
    r : int, default 2
        Number of latent factors.

    Returns
    -------
    RichResult
        keys: ``att``, ``gap`` (T,), ``y0_hat`` (T,), ``loadings``
        (r,), ``r``, ``treat_time``, ``method``.

    References
    ----------
    Xu, Y. (2017). Generalized synthetic control method: causal
    inference with interactive fixed effects models. *Political
    Analysis*, 25(1), 57-76. doi:10.1017/pan.2016.2.
    """
    y1 = np.asarray(y_treated, dtype=float).ravel()
    Y0 = np.asarray(y_controls, dtype=float)
    if Y0.ndim != 2 or Y0.shape[0] != y1.size:
        raise ValueError("y_controls must be (T, J) matching y_treated.")
    T, J = Y0.shape
    t0 = int(treat_time)
    if not 2 <= t0 < T:
        raise ValueError(f"treat_time must lie in [2, T), got {t0}.")
    r = int(r)
    if not 1 <= r <= min(t0 - 1, J - 1):
        raise ValueError(f"r must lie in [1, min(t0 - 1, J - 1)] = [1, {min(t0 - 1, J - 1)}].")

    # factors from the control panel (unit-demeaned), PCA over time
    mu_t = Y0.mean(axis=1)  # common time effect
    Z = Y0 - mu_t[:, None]
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    F = U[:, :r] * s[:r]  # (T, r) estimated factors

    # treated loadings from the pre period
    D = np.column_stack([np.ones(t0), F[:t0]])
    coef, *_ = np.linalg.lstsq(D, y1[:t0] - mu_t[:t0], rcond=None)
    y0_hat = mu_t + np.column_stack([np.ones(T), F]) @ coef
    gap = y1 - y0_hat

    return RichResult(
        payload={
            "att": float(gap[t0:].mean()),
            "gap": gap,
            "y0_hat": y0_hat,
            "loadings": coef[1:],
            "r": r,
            "treat_time": t0,
            "method": "Generalized Synthetic Control (interactive fixed effects)",
        }
    )


def cheatsheet():
    return "gscmcl: PCA factors from donors, treated loadings from pre period, impute Y(0)"
