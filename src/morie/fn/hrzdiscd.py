# morie.fn -- function file (rootcoder007/morie)
"""Direct estimation of single-index model with discrete covariates."""

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_direct_discrete_x"]


def horowitz_direct_discrete_x(x, y, z=None, beta=None, c0=None, c1=None,
                               bandwidth=None, n_grid=60):
    r"""Direct estimation of a single-index model with both
    continuous and DISCRETE covariates (Horowitz Sec. 2.6.3),
    equations (2.45)-(2.51):

    .. math:: E(Y \mid X = x, Z = z) = G(x'\beta + z'\alpha),

    with X continuous and Z discrete.

    Average-derivative methods CANNOT estimate :math:`\alpha`.
    Derivatives of :math:`E(Y|X = x, Z = z)` with respect to the
    discrete components do not exist, so no amount of averaging
    produces them -- that absence, not a technical inconvenience, is
    why the section exists.

    The construction goes around it. :math:`\beta` is estimated
    stratum by stratum, running an average-derivative estimator
    separately on each value :math:`z^{(i)}` of Z and combining by
    (2.46),

    .. math:: b_n = \frac{\sum_i w_{ni}\delta_n^{(i)}}
                         {\sum_i w_{ni}\delta_{n1}^{(i)}},

    which is a weighted average renormalised by its first component.
    :math:`\alpha` then comes from a LINEAR system. Under the weak
    monotonicity of Assumption G, the functional

    .. math:: J(z) = \int_{v_0}^{v_1}\big\{c_0\mathbf 1[G(v + z'\alpha) < c_0]
              + c_1\mathbf 1[G(v + z'\alpha) > c_1]
              + G(v + z'\alpha)\mathbf 1[c_0 \le G \le c_1]\big\}dv

    satisfies :math:`J[z^{(i)}] - J[z^{(1)}] =
    (c_1 - c_0)(z^{(i)} - z^{(1)})'\alpha` (2.47), which is
    :math:`M - 1` linear equations in :math:`\alpha`, solved by

    .. math:: \alpha = (c_1 - c_0)(W'W)^{-1}W'\Delta J. \tag{2.48}

    So a shift in the discrete covariate shows up as a horizontal
    SHIFT of G, and integrating the truncated G recovers the shift
    size. Identification needs at least one continuous covariate,
    and needs :math:`W'W` nonsingular -- with :math:`M - 1` less than
    the number of discrete coefficients there is nothing to solve,
    and that is reported rather than silently pseudo-inverted.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Continuous covariates.
    y : array-like, shape (n,)
        Response.
    z : array-like, shape (n,) or (n, dz), optional
        Discrete covariates. Without them this reduces to an
        ordinary average-derivative estimate of beta.
    beta : array-like, optional
        A root-n-consistent estimate of the continuous coefficients.
        Supplied, it is used directly and only alpha is estimated;
        omitted, beta comes from the stratum-wise construction
        (2.46). Both routes are offered because a caller who has
        already run one of the Sec. 2.5-2.6 estimators has no reason
        to re-estimate beta here.
    c0, c1 : float, optional
        The truncation levels of Assumption G, with
        :math:`c_0 < c_1`. Default to the 20th and 80th percentiles
        of the fitted regression.
    bandwidth : float, optional
        Bandwidth for the within-stratum regressions.
    n_grid : int, default 60
        Points used for the integral defining J.

    Returns
    -------
    RichResult
        keys: ``beta``, ``alpha``, ``support_z``, ``delta_by_stratum``,
        ``weights``, ``J``, ``c0``, ``c1``, ``identified``,
        ``average_derivative_can_estimate_alpha`` (False),
        ``n``, ``d``, ``dz``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 2.6.3 (direct estimation with
    discrete covariates), eqs. (2.45)-(2.51), Assumption G and
    Theorem 2.5; Horowitz and Hardle (1996).
    """
    from ._horowitz import nw_regression, silverman_bw
    from .hrzade import hrz_average_derivative

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    n, d = X.shape
    if n < 20:
        raise ValueError(f"need at least 20 observations, got {n}.")
    if d < 1:
        raise ValueError("identification requires at least one continuous "
                         "covariate.")

    def _normalise(vec):
        v = np.atleast_1d(np.asarray(vec, dtype=float)).ravel()
        if v.size != d:
            raise ValueError(f"beta has {v.size} entries for {d} covariates.")
        if v[0] == 0:
            raise ValueError("the scale normalisation needs a nonzero first "
                             "coefficient.")
        return v / abs(v[0])

    if z is None:
        if beta is None:
            delta = np.atleast_1d(hrz_average_derivative(X, yv)["delta"])
            b = delta / abs(delta[0])
        else:
            b = _normalise(beta)
        return RichResult(payload={
            "beta": b, "alpha": None, "support_z": None,
            "delta_by_stratum": None, "weights": None, "J": None,
            "beta_source": "supplied" if beta is not None else "average derivative",
            "c0": None, "c1": None, "identified": True,
            "average_derivative_can_estimate_alpha": False,
            "n": int(n), "d": int(d), "dz": 0,
            "method": "No discrete covariates: this is the plain average-derivative estimate"})

    Z = np.atleast_2d(np.asarray(z, dtype=float))
    if Z.shape[0] != n:
        Z = Z.T
    if Z.shape[0] != n:
        raise ValueError("z must have one row per entry of y.")
    dz = Z.shape[1]
    support, inverse = np.unique(Z, axis=0, return_inverse=True)
    M = support.shape[0]
    if M < 2:
        raise ValueError("z takes a single value; alpha is not identified.")

    # (2.46): stratum-wise average derivatives, weighted by stratum size.
    # A caller who already has a root-n estimate of beta -- from any of
    # the estimators in Sections 2.5-2.6 -- can supply it instead and
    # skip straight to alpha; the section's point is that alpha cannot
    # be obtained this way, not that beta cannot.
    if beta is None:
        deltas, wn = [], []
        for m in range(M):
            sel = inverse == m
            nm = int(sel.sum())
            if nm < 10:
                raise ValueError(
                    f"stratum {m} has {nm} observations, too few for an "
                    "average-derivative estimate.")
            deltas.append(np.atleast_1d(
                hrz_average_derivative(X[sel], yv[sel])["delta"]))
            wn.append(nm / n)
        deltas = np.array(deltas)
        wn = np.array(wn)
        num = (wn[:, None] * deltas).sum(axis=0)
        den = float((wn * deltas[:, 0]).sum())
        if den == 0:
            raise ValueError("the weighted first component of the stratum "
                             "average derivatives is zero; beta is not "
                             "normalisable.")
        b = num / den
        beta_source = "stratum-wise (2.46)"
    else:
        b = _normalise(beta)
        deltas = wn = None
        beta_source = "supplied"

    v = X @ b
    hh = float(silverman_bw(v)) if bandwidth is None else float(bandwidth)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")
    fitted = nw_regression(v, yv, grid=v, h=hh)[1]
    cc0 = float(np.quantile(fitted, 0.2)) if c0 is None else float(c0)
    cc1 = float(np.quantile(fitted, 0.8)) if c1 is None else float(c1)
    if cc0 >= cc1:
        raise ValueError(f"need c0 < c1, got {(cc0, cc1)}.")

    # common integration range over the index
    v0, v1 = np.quantile(v, [0.1, 0.9])
    grid = np.linspace(v0, v1, int(n_grid))
    J = np.empty(M)
    for m in range(M):
        sel = inverse == m
        gm = nw_regression(v[sel], yv[sel], grid=grid, h=hh)[1]
        trunc = np.where(gm < cc0, cc0, np.where(gm > cc1, cc1, gm))
        J[m] = float(np.trapezoid(trunc, grid))

    # (2.48): alpha from the M-1 linear equations
    W = support[1:] - support[0]
    dJ = J[1:] - J[0]
    WtW = W.T @ W
    identified = bool(np.linalg.matrix_rank(WtW) == dz)
    alpha = (np.linalg.solve(WtW, W.T @ dJ) / (cc1 - cc0)) if identified else None

    return RichResult(payload={
        "beta": b, "alpha": alpha, "support_z": support,
        "delta_by_stratum": deltas, "weights": wn, "J": J,
        "beta_source": beta_source,
        "c0": cc0, "c1": cc1, "identified": identified,
        "average_derivative_can_estimate_alpha": False,
        "n": int(n), "d": int(d), "dz": int(dz),
        "method": "(2.46) for beta stratum-wise; (2.48) for alpha, since derivatives in z do not exist"})


def cheatsheet():
    return "hrzdiscd: no derivative in a discrete covariate exists -- alpha comes from a LINEAR system"
