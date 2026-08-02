# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic properties of Horowitz's T_n and F_n estimators."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_T_F_asymp_props"]


def horowitz_T_F_asymp_props(x, y, bandwidth, n=None):
    r"""Asymptotic properties of Horowitz's :math:`T_n` and
    :math:`F_n` (Horowitz Sec. 6.3.2), Theorems 6.4 and 6.5:

    .. math:: \operatorname*{plim}_{n\to\infty}
              \sup_{y_2 \le y \le y_1} |T_n(y) - T(y)| = 0,
              \qquad
              n^{1/2}(T_n - T) \Rightarrow \text{a tight, mean-zero
              Gaussian process,}

    and the same pair for :math:`F_n` over :math:`[u_0, u_1]`.

    Two things are easy to state loosely and are stated precisely
    here instead:

    * the limit is a GAUSSIAN PROCESS, not a normal distribution.
      :math:`T_n` is a random FUNCTION; convergence is weak
      convergence in a function space, and the covariance function
      -- given in Horowitz (1996), not reproduced in the book -- is
      what any confidence band needs. A pointwise normal
      approximation is a consequence, not the theorem.
    * consistency and root-n normality hold only on a compact
      interval strictly inside the support, because T may be
      unbounded at the boundaries.

    Assumption HT9 constrains the two bandwidths differently, and
    that asymmetry is the practical content: with :math:`K_Y` of
    second order and :math:`K_Z` of SIXTH order,

    .. math:: h_{ny} \propto n^{-1/3}, \qquad h_{nz} \propto n^{-1/10}.

    :math:`h_{nz}` shrinks far more slowly than any density-estimation
    rule would suggest, because :math:`G_{nz}` is a functional of
    DERIVATIVES of :math:`K_Z` and those converge slowly. Using one
    bandwidth for both, or a second-order :math:`K_Z`, breaks the
    theorem rather than merely costing efficiency.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates; used for the sample size and dimension.
    y : array-like, shape (n,)
        Response.
    bandwidth : float or pair of floats
        The ``(h_ny, h_nz)`` actually used, checked against HT9.
    n : int, optional
        Sample size to report the rates at; taken from the data
        otherwise.

    Returns
    -------
    RichResult
        keys: ``asymptotic_distribution``, ``limit_is_process``
        (True), ``rate``, ``rate_exponent`` (-1/2),
        ``h_ny_reference``, ``h_nz_reference``, ``h_ny``, ``h_nz``,
        ``bandwidths_consistent_with_HT9``, ``Kz_order_required`` (6),
        ``uniform_over``, ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 6.3.2, assumptions HT1-HT9 and
    Theorems 6.4-6.5; Horowitz (1996).
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    nn = int(X.shape[0]) if n is None else int(n)
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    hb = np.atleast_1d(np.asarray(bandwidth, dtype=float)).ravel()
    h_ny, h_nz = (float(hb[0]), float(hb[0])) if hb.size == 1 else \
        (float(hb[0]), float(hb[1]))
    if h_ny <= 0 or h_nz <= 0:
        raise ValueError(f"bandwidths must be positive, got {(h_ny, h_nz)}.")

    ref_y = float(nn ** (-1.0 / 3.0))
    ref_z = float(nn ** (-1.0 / 10.0))
    # HT9 fixes the RATES, not the constants, so the check is on
    # order of magnitude rather than on equality
    ok = bool(0.1 <= h_ny / ref_y <= 10.0 and 0.1 <= h_nz / ref_z <= 10.0
              and h_nz > h_ny)
    return RichResult(payload={
        "asymptotic_distribution": "tight mean-zero Gaussian process",
        "limit_is_process": True,
        "rate": float(nn ** -0.5), "rate_exponent": -0.5,
        "h_ny_reference": ref_y, "h_nz_reference": ref_z,
        "h_ny": h_ny, "h_nz": h_nz,
        "bandwidths_consistent_with_HT9": ok,
        "Kz_order_required": 6,
        "uniform_over": "a compact interval strictly inside the support of Y",
        "n": nn, "d": int(X.shape[1]),
        "method": "Theorems 6.4-6.5: uniform consistency and n^{1/2} weak convergence to a Gaussian PROCESS"})


def cheatsheet():
    return "hrztfap: the limit is a Gaussian PROCESS; h_nz ~ n^{-1/10} is far slower than h_ny ~ n^{-1/3}"
