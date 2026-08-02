# morie.fn -- function file (rootcoder007/morie)
"""Panel data deconvolution: estimate fU and f_eps from Y_jt = X_jt'beta + U_j + eps_jt."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_panel_deconvolution", "horowitz_panel_deconv"]


def horowitz_panel_deconvolution(y, x, beta, nu_U=None, nu_eps=None,
                                 grid_u=None, grid_z=None):
    r"""Nonparametric densities of the individual effect and the
    idiosyncratic error in a panel model (Horowitz Sec. 5.2.1),
    equations (5.21)-(5.26):

    .. math:: Y_{jt} = X_{jt}'\beta + U_j + \varepsilon_{jt}.

    Two residual sets do the work. The levels residual
    :math:`W_{njt} = Y_{jt} - b_n'X_{jt}` estimates
    :math:`W = U + \varepsilon`, so :math:`\psi_W = \psi_U\psi_
    \varepsilon`. The within-individual difference
    :math:`\eta_{njt} = (Y_{jt} - Y_{j1}) - b_n'(X_{jt} - X_{j1})`
    removes :math:`U_j` completely and estimates the difference of
    two independent copies of :math:`\varepsilon`, so
    :math:`\psi_\eta = |\psi_\varepsilon|^2`. Hence

    .. math:: \psi_\varepsilon = \psi_\eta^{1/2},
              \qquad
              \psi_U = \frac{\psi_W}{\psi_\eta^{1/2}}.

    **The square root is what the symmetry assumption is for.**
    :math:`\psi_\eta = |\psi_\varepsilon|^2` determines
    :math:`\psi_\varepsilon` only up to sign; assuming
    :math:`\varepsilon` symmetric about zero makes
    :math:`\psi_\varepsilon` real, and assuming it never vanishes
    makes it positive, which picks the root. Without symmetry this
    construction does not identify :math:`f_\varepsilon`, even though
    symmetry is not needed for identification in general.

    This is a HARDER problem than Sec. 5.1. There the contaminating
    distribution was known; here the distribution of
    :math:`\varepsilon` is unknown and must be recovered from
    :math:`\eta` first. The rates are just as slow: for normal
    :math:`\varepsilon` and twice-differentiable densities the
    fastest possible rate is :math:`(\log n)^{-1}`.

    Asymptotics are in n with T FIXED, because panels are typically
    wide and short.

    Parameters
    ----------
    y : array-like, shape (n, T)
        Responses.
    x : array-like, shape (n, T, d) or (n*T, d)
        Covariates.
    beta : array-like, shape (d,)
        A root-n-consistent estimate of beta.
    nu_U, nu_eps : float, optional
        Smoothing bandwidths for (5.26) and (5.25). Defaults follow
        the amplification criterion, ``(log n)**(-1/2)``, matching
        the logarithmic rate the theory allows.
    grid_u, grid_z : array-like, optional
        Evaluation points for :math:`f_U` and :math:`f_\varepsilon`.

    Returns
    -------
    RichResult
        keys: ``grid_u``, ``f_U``, ``grid_z``, ``f_eps``,
        ``psi_eps_from_root`` (True), ``symmetry_required`` (True),
        ``nu_U``, ``nu_eps``, ``fastest_possible_rate``,
        ``asymptotics_in`` ("n with T fixed"), ``n``, ``T``, ``d``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 5.2.1-5.2.2, eqs. (5.21)-(5.26)
    and Theorem 5.4; Horowitz and Markatou (1996).
    """
    from ._hrz_paneldec import deconvolve_pair, panel_residuals

    Y = np.atleast_2d(np.asarray(y, dtype=float))
    n, T = Y.shape
    b = np.asarray(beta, dtype=float).ravel()
    W, eta = panel_residuals(Y, x, b)
    if n < 10:
        raise ValueError(f"need at least 10 individuals, got {n}.")

    default = float(np.log(n) ** -0.5)
    nU = default if nu_U is None else float(nu_U)
    ne = default if nu_eps is None else float(nu_eps)
    gu = np.linspace(np.quantile(W, 0.05), np.quantile(W, 0.95), 61) \
        if grid_u is None else np.atleast_1d(np.asarray(grid_u, dtype=float))
    gz = np.linspace(np.quantile(eta, 0.05), np.quantile(eta, 0.95), 61) \
        if grid_z is None else np.atleast_1d(np.asarray(grid_z, dtype=float))

    f_U, f_eps = deconvolve_pair(W, eta, gu, gz, nU, ne)
    return RichResult(payload={
        "grid_u": gu, "f_U": f_U, "grid_z": gz, "f_eps": f_eps,
        "psi_eps_from_root": True, "symmetry_required": True,
        "nu_U": nU, "nu_eps": ne,
        "fastest_possible_rate": "(log n)^{-1} for normal eps",
        "asymptotics_in": "n with T fixed",
        "n": int(n), "T": int(T), "d": int(b.size),
        "method": "Panel deconvolution (5.21)-(5.26); psi_eps = psi_eta^{1/2} needs eps symmetric"})


def cheatsheet():
    return "hrzpanel: differencing kills U_j so eta identifies eps; the root needs SYMMETRY"


#: Catalogue alias for :func:`horowitz_panel_deconvolution`.
horowitz_panel_deconv = horowitz_panel_deconvolution
