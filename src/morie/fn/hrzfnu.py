# morie.fn -- function file (rootcoder007/morie)
"""Smoothed deconvolution estimator of fU."""

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_smoothed_fU", "horowitz_deconv_estimator"]


def horowitz_smoothed_fU(y, x, beta, nu_U=None, grid=None):
    r"""The smoothed estimator of :math:`f_U` alone (Horowitz
    Sec. 5.2.1), equation (5.26):

    .. math:: f_{nU}(u) = \frac{1}{2\pi}\int_{-\infty}^{\infty}
              e^{-i\tau u}\,
              \frac{\psi_{nW}(\tau)\,\psi_\zeta(\nu_{nU}\tau)}
                   {|\psi_{n\eta}(\tau)|^{1/2}}\, d\tau.

    Substituting the empirical characteristic functions straight into
    the inversion formula does NOT work -- the resulting integral
    does not exist in general, because
    :math:`\psi_{n\eta}` decays while the numerator does not, and the
    ratio need not be integrable. :math:`\psi_\zeta` is the
    regularisation: a characteristic function supported on
    :math:`[-1, 1]`, so the integrand is identically zero outside
    :math:`|\tau| \le 1/\nu_{nU}` and the ratio is never formed where
    the denominator has died. It is the Fourier-transform analogue of
    kernel smoothing, and it is mandatory rather than a refinement.

    Parameters
    ----------
    y : array-like, shape (n, T)
        Panel responses.
    x : array-like, shape (n, T, d) or (n*T, d)
        Covariates.
    beta : array-like, shape (d,)
        Root-n-consistent beta.
    nu_U : float, optional
        Smoothing bandwidth; ``(log n)**(-1/2)`` otherwise.
    grid : array-like, optional
        Evaluation points.

    Returns
    -------
    RichResult
        keys: ``grid``, ``f_U``, ``nu_U``, ``cutoff``
        (:math:`1/\nu_{nU}`), ``regularisation_required`` (True),
        ``n``, ``T``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 5.2.1, eq. (5.26).
    """
    from ._hrz_paneldec import deconvolve_pair, panel_residuals

    Y = np.atleast_2d(np.asarray(y, dtype=float))
    n, T = Y.shape
    b = np.asarray(beta, dtype=float).ravel()
    W, eta = panel_residuals(Y, x, b)
    if n < 10:
        raise ValueError(f"need at least 10 individuals, got {n}.")
    nU = float(np.log(n) ** -0.5) if nu_U is None else float(nu_U)
    if nU <= 0:
        raise ValueError(f"nu_U must be positive, got {nU}.")
    g = np.linspace(np.quantile(W, 0.05), np.quantile(W, 0.95), 61) \
        if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    f_U, _ = deconvolve_pair(W, eta, g, g[:1], nU, nU)
    return RichResult(payload={
        "grid": g, "f_U": f_U, "nu_U": nU, "cutoff": 1.0 / nU,
        "regularisation_required": True,
        "n": int(n), "T": int(T),
        "method": "(5.26): psi_zeta compactly supported, so the ratio is never formed past the cut-off"})


def cheatsheet():
    return "hrzfnu: without psi_zeta the inversion integral does not exist -- regularisation is mandatory"


#: Catalogue alias for :func:`horowitz_smoothed_fU`.
horowitz_deconv_estimator = horowitz_smoothed_fU
