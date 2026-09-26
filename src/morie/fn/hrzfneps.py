# morie.fn -- function file (rootcoder007/morie)
"""Estimators fn_eps and fn_U for panel deconvolution."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["horowitz_fn_eps_fn_U", "horowitz_panel_density_estimators"]


def horowitz_fn_eps_fn_U(y, x, beta, nu_U=None, nu_eps=None,
                         grid_u=None, grid_z=None, kernel="fourfold"):
    r"""Both smoothed panel-deconvolution estimators together
    (Horowitz Sec. 5.2.1-5.2.2), equations (5.25) and (5.26):

    .. math:: f_{n\varepsilon}(z) = \frac{1}{2\pi}\int e^{-i\tau z}
              |\psi_{n\eta}(\tau)|^{1/2}\psi_\zeta(\nu_{n\varepsilon}
              \tau)\,d\tau,

    .. math:: f_{nU}(u) = \frac{1}{2\pi}\int e^{-i\tau u}
              \frac{\psi_{nW}(\tau)\psi_\zeta(\nu_{nU}\tau)}
                   {|\psi_{n\eta}(\tau)|^{1/2}}\,d\tau.

    The pair is returned together because the two bandwidths are NOT
    interchangeable and Theorem 5.4 treats them separately:

    .. math:: \sup_z |f_{n\varepsilon} - f_\varepsilon|
              = O_p(\nu_{n\varepsilon}^2) + O_p(B_{n\varepsilon}/
              \nu_{n\varepsilon}) + o_p(A_{n\varepsilon}/
              \nu_{n\varepsilon}),

    and likewise for :math:`f_{nU}` in :math:`\nu_{nU}`. The two
    error terms pull in opposite directions -- the
    :math:`\nu^2` term is bias and shrinks with the bandwidth, the
    :math:`B_n/\nu` term is variance and grows -- which is the usual
    trade-off, except that :math:`A_n` and :math:`B_n` carry
    :math:`\psi_\zeta(1/\nu)^{-2}`, so the variance term explodes far
    faster than in ordinary density estimation. Assumption P4 is what
    keeps both under control.

    :math:`f_{n\varepsilon}` needs no division at all -- it is built
    from :math:`|\psi_{n\eta}|^{1/2}` directly -- while
    :math:`f_{nU}` divides by it. That asymmetry is why the two get
    different bandwidths in practice, and both are returned so the
    difference is visible.

    Parameters
    ----------
    y : array-like, shape (n, T)
        Panel responses.
    x : array-like, shape (n, T, d) or (n*T, d)
        Covariates.
    beta : array-like, shape (d,)
        Root-n-consistent beta.
    nu_U, nu_eps : float, optional
        The two bandwidths; by default ``sigma_eps / sqrt(log n)`` and
        ``0.5 sigma_eps N**(-1/5)`` (``_hrz_paneldec.default_bandwidths``).
    grid_u, grid_z : array-like, optional
        Evaluation points.
    kernel : {"fourfold", "flattop"}, default "fourfold"
        ``psi_zeta``. "fourfold" is the book's characteristic function
        (assumption PHU7); "flattop" is the indicator of [-1, 1], outside
        PHU7 but with higher-order bias and 2-3 times smaller errors in
        simulation. Each has its own default bandwidths.

    Returns
    -------
    RichResult
        keys: ``grid_u``, ``f_U``, ``grid_z``, ``f_eps``, ``nu_U``,
        ``nu_eps``, ``f_eps_requires_division`` (False),
        ``f_U_requires_division`` (True), ``bandwidths_independent``
        (True), ``n``, ``T``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 5.2.1-5.2.2, eqs. (5.25)-(5.26),
    assumptions P1-P4 and Theorem 5.4.
    """
    from ._hrz_paneldec import _check_kernel, deconvolve_pair, default_bandwidths, panel_residuals
    _check_kernel(kernel)
    Y = np.atleast_2d(np.asarray(y, dtype=float))
    n, T = Y.shape
    b = np.asarray(beta, dtype=float).ravel()
    W, eta = panel_residuals(Y, x, b)
    if n < 10:
        raise ValueError(f"need at least 10 individuals, got {n}.")
    dU, dE = default_bandwidths(eta, n, kernel, W)
    nU = dU if nu_U is None else float(nu_U)
    ne = dE if nu_eps is None else float(nu_eps)
    gu = np.linspace(np.quantile(W, 0.05), np.quantile(W, 0.95), 61) \
        if grid_u is None else np.atleast_1d(np.asarray(grid_u, dtype=float))
    gz = np.linspace(np.quantile(eta, 0.05), np.quantile(eta, 0.95), 61) \
        if grid_z is None else np.atleast_1d(np.asarray(grid_z, dtype=float))
    f_U, f_eps = deconvolve_pair(W, eta, gu, gz, nU, ne, kernel=kernel)
    return RichResult(payload={
        "kernel": kernel,
        "grid_u": gu, "f_U": f_U, "grid_z": gz, "f_eps": f_eps,
        "nu_U": nU, "nu_eps": ne,
        "f_eps_requires_division": False,
        "f_U_requires_division": True,
        "bandwidths_independent": True,
        "n": int(n), "T": int(T),
        "method": "(5.25) needs no division; (5.26) does, so the two carry separate bandwidths"})


def cheatsheet():
    return "hrzfneps: f_eps never divides by the root, f_U does -- hence separate bandwidths"


#: Catalogue alias for :func:`horowitz_fn_eps_fn_U`.
horowitz_panel_density_estimators = horowitz_fn_eps_fn_U
