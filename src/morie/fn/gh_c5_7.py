# morie.fn -- function file (rootcoder007/morie)
"""Predictive recursion deconvolution algorithm for DPM estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_pred_rec"]


def ghosal_pred_rec(x, theta_grid=None, sigma=1.0, weights=None, f0=None):
    r"""Newton's predictive recursion (Ghosal Sec. 5.4):

    .. math:: \hat f_i(\theta) = (1-w_i)\hat f_{i-1}(\theta)
              + w_i \frac{\psi(X_i;\theta)\hat f_{i-1}(\theta)}
                          {\int \psi(X_i;t)\hat f_{i-1}(t)dt},

    estimating the MIXING density in ``p_F(x) = int psi(x;theta)dF``
    -- a deconvolution, since the observed density is a smear of the
    thing wanted.

    A single sweep through the data, no MCMC and no iteration to
    convergence, which makes it enormously cheaper than fitting a DP
    mixture properly. The price is stated rather than hidden: the
    estimate DEPENDS ON THE ORDER of the observations, because each
    step conditions on the running estimate. ``order_dependent`` is
    returned as True, and permuting the sample genuinely changes the
    answer.

    The weights need ``sum w_i = infinity`` and
    ``sum w_i^2 < infinity``; ``w_i = (i+2)^{-2/3}`` satisfies both
    and is the default; the more obvious ``(i+1)^{-2/3}`` starts at
    ``w = 1``, which discards the initial estimate rather than
    combining with it.

    Parameters
    ----------
    x : array-like
        Observations from the mixture.
    theta_grid : array-like, optional
        Support of the mixing density.
    sigma : float > 0
        Scale of the Gaussian mixing kernel.
    weights : array-like, optional
        The ``w_i``; must lie strictly in (0, 1).
    f0 : array-like, optional
        Initial mixing density; uniform otherwise.

    Returns
    -------
    RichResult
        keys: ``theta_grid``, ``f_mixing``, ``mixed_density``,
        ``order_dependent`` (True), ``single_pass`` (True),
        ``weight_rule``, ``n``, ``method``.
    References
    ----------
    Ghosal and van der Vaart, Sec. 5.4 (predictive recursion
    deconvolution algorithm); Newton (2002).
    """
    from ._ghosal import predictive_recursion

    xv = np.asarray(x, dtype=float).ravel()
    if xv.size < 2:
        raise ValueError(f"need at least 2 observations, got {xv.size}.")
    s = float(sigma)
    if s <= 0:
        raise ValueError(f"sigma must be positive, got {s}.")
    th = np.linspace(xv.min() - 2 * s, xv.max() + 2 * s, 201) \
        if theta_grid is None else \
        np.atleast_1d(np.asarray(theta_grid, dtype=float))

    def kern(xi, t):
        return np.exp(-0.5 * ((xi - t) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    f = predictive_recursion(xv, th, kern, f0=f0, weights=weights)
    mixed = np.array([float(np.trapezoid(kern(v, th) * f, th)) for v in th])
    return RichResult(payload={
        "theta_grid": th, "f_mixing": f, "mixed_density": mixed,
        "order_dependent": True, "single_pass": True,
        "weight_rule": "w_i = (i+2)^{-2/3}: sum w = inf, sum w^2 < inf, and w_1 < 1",
        "n": int(xv.size),
        "method": "Predictive recursion (Sec. 5.4); one sweep, no MCMC, order dependent"})


def cheatsheet():
    return "gh_c5_7: one pass, no MCMC -- and the answer depends on the ORDER of the data"


# compact alias per ledger/NAMING.md
ghosalpredrec = ghosal_pred_rec
