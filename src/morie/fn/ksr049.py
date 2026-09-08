# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic linearity of a Z-estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_asymptotic_linearity", "kosorok_ch2_z_master_linearization"]


def kosorok_asymptotic_linearity(psi_dot, psi_n, psi, theta_n, theta0, n,
                                 grid=None):
    r"""Asymptotic linearity of a Z-estimator (Kosorok Eq. 2.13,
    p. 26):

    .. math:: \big\|\sqrt n\,\dot\Psi_{\theta_0}
              (\theta_n - \theta_0)
              + \sqrt n(\Psi_n - \Psi)(\theta_0)\big\|_L
              \to 0.

    The payoff of Chapter 2. It says
    :math:`\sqrt n(\theta_n - \theta_0)` behaves like
    :math:`-\dot\Psi_{\theta_0}^{-1}\sqrt n(\Psi_n -
    \Psi)(\theta_0)`: an ESTIMATOR, which is an implicit function
    of the data, is replaced by a linear functional of an empirical
    process, whose limit is known. Everything after -- weak
    convergence, the bootstrap, efficiency -- follows from this one
    representation.

    The derivative :math:`\dot\Psi_{\theta_0}` must be
    continuously invertible, and that is a genuine condition rather
    than a formality: where it fails the estimator is typically not
    root-n at all. ``derivative_invertible`` reports it.

    Parameters
    ----------
    psi_dot : array-like or callable
        The derivative operator at ``theta0``.
    psi_n, psi : callable
        ``(theta, t)`` maps.
    theta_n, theta0 : array-like
        The estimate and the truth.
    n : int
        Sample size.
    grid : array-like, optional
        Points for the uniform norm.

    Returns
    -------
    RichResult
        keys: ``residual_norm``, ``linear_term``, ``process_term``,
        ``derivative_invertible``, ``implies``, ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2, Eq. (2.13), p. 26.
    """
    g = np.linspace(0.0, 1.0, 51) if grid is None else \
        np.atleast_1d(np.asarray(grid, dtype=float))
    nn = int(n)
    if nn < 1:
        raise ValueError(f"n must be at least 1, got {nn}.")
    th = np.atleast_1d(np.asarray(theta_n, dtype=float)).ravel()
    t0 = np.atleast_1d(np.asarray(theta0, dtype=float)).ravel()
    if th.size != t0.size:
        raise ValueError("theta_n and theta0 must have the same length.")
    D = np.atleast_2d(np.asarray(psi_dot(t0) if callable(psi_dot) else psi_dot,
                                 dtype=float))
    if D.shape[1] != th.size:
        raise ValueError(f"psi_dot must have {th.size} columns.")
    inv_ok = bool(np.linalg.matrix_rank(D) == min(D.shape))
    lin = np.sqrt(nn) * (D @ (th - t0))
    proc = np.array([float(psi_n(t0, v) - psi(t0, v)) for v in g]) * np.sqrt(nn)
    # Eq. (2.13) takes the norm of Psi_dot(theta_n - theta_0) PLUS the
    # process, both as elements of the SAME space L. Psi_dot must
    # therefore map into L: one row (a scalar functional, broadcast
    # against the grid) or one row per grid point. Summing the
    # components of a p-vector -- the old behaviour -- paired terms
    # from different spaces and was wrong for p > 1.
    if lin.size == 1:
        resid = float(np.max(np.abs(lin[0] + proc)))
    elif lin.size == proc.size:
        resid = float(np.max(np.abs(lin + proc)))
    else:
        raise ValueError(
            "psi_dot must map into the same space as the process: give it "
            f"1 or {proc.size} rows (grid points), got {lin.size}.")
    return RichResult(payload={
        "residual_norm": resid, "linear_term": lin, "process_term": proc,
        "derivative_invertible": inv_ok,
        "implies": "sqrt(n)(theta_n - theta_0) ~ -Psi_dot^{-1} sqrt(n)(Psi_n - Psi)(theta_0)",
        "n": nn,
        "method": "Asymptotic linearity (Eq. 2.13); an estimator becomes a linear functional of a process"})


def cheatsheet():
    return "ksr049: this one representation is what everything after Chapter 2 is built on"


#: Catalogue alias for :func:`kosorok_asymptotic_linearity`.
kosorok_ch2_z_master_linearization = kosorok_asymptotic_linearity
