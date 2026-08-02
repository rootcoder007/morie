# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric rank estimator for single-index model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_semipar_rank"]


def horowitz_semipar_rank(x, y, variant="mrc", M=None, n_restarts=8, seed=0):
    r"""Maximum rank correlation and Cavanagh-Sherman estimators of a
    single-index model (Horowitz Sec. 2.5.5):

    .. math:: b_{n,MRC} = \arg\max_b \frac{1}{n(n-1)}
              \sum_i\sum_{j\ne i}
              \mathbf 1\{Y_i > Y_j\}\,\mathbf 1\{X_i'b > X_j'b\},

    .. math:: b_{n,CS} = \arg\max_b \frac{1}{n(n-1)}
              \sum_i\sum_{j\ne i} M(Y_i)\,
              \mathbf 1\{X_i'b > X_j'b\},

    with M increasing.

    The identifying observation is an ordering, not a moment: if G is
    NONDECREASING and :math:`Y - G(X'\beta)` is independent of X,
    then :math:`X_i'\beta > X_j'\beta` implies
    :math:`P(Y_i > Y_j) > P(Y_j > Y_i)`. So the estimator simply
    makes the rank ordering of the Y's match that of the indices as
    closely as possible; G never appears.

    That is exactly the appeal and exactly the cost:

    * **no bandwidth.** These estimators need no kernel, no
      smoothing parameter and no nonparametric pre-estimate of G.
      In a chapter where every other estimator turns on a bandwidth
      choice, that is a real practical advantage.
    * **not asymptotically efficient**, and the book says so plainly.
      They are :math:`n^{-1/2}`-consistent and asymptotically normal
      (Sherman 1993; Cavanagh and Sherman 1998), but they do not
      attain :math:`\Omega_{SI}`.
    * **inference is awkward.** The asymptotic covariance matrices
      are hard to implement; the practical route is the bootstrap,
      which Subbotin (2008) proves is consistent here. No analytic
      standard error is returned, because a plausible-looking one
      would be the wrong thing to trust.

    The Cavanagh-Sherman variant is consistent under WEAKER
    conditions than MRC and is easier to compute -- its objective
    drops the :math:`\mathbf 1\{Y_i > Y_j\}` factor, so the double
    sum separates.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates; the first column carries the scale normalisation.
    y : array-like, shape (n,)
        Response; only its ORDERING is used.
    variant : {"mrc", "cs"}
        Han's maximum rank correlation, or Cavanagh-Sherman.
    M : callable, optional
        The increasing function for the CS variant; the rank of y
        otherwise.
    n_restarts : int, default 8
        Restarts, since the objective is a step function.
    seed : int, default 0
        RNG seed for the restarts.

    Returns
    -------
    RichResult
        keys: ``beta``, ``objective``, ``variant``,
        ``requires_bandwidth`` (False),
        ``asymptotically_efficient`` (False), ``rate_exponent``
        (-1/2), ``inference`` ("bootstrap"), ``se`` (None), ``n``,
        ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 2.5.5 (semiparametric rank
    estimators); Han (1987), Sherman (1993), Cavanagh and Sherman
    (1998), Subbotin (2008).
    """
    from ._horowitz import (GRID_SCAN_HALF_WIDTH, GRID_SCAN_POINTS,
                            optimize_scale_normalized)

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    if variant not in ("mrc", "cs"):
        raise ValueError("variant must be 'mrc' or 'cs'.")
    n, d = X.shape
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")
    if d < 2:
        raise ValueError(f"need at least 2 covariates, got {d}.")

    denom = n * (n - 1)
    if variant == "mrc":
        gt = (yv[:, None] > yv[None, :]).astype(float)
        np.fill_diagonal(gt, 0.0)

        def score(b):
            z = X @ b
            return float(np.sum(gt * (z[:, None] > z[None, :]))) / denom
    else:
        if M is None:
            # the ranks of y: increasing in y, which is all CS needs
            mv = np.argsort(np.argsort(yv)).astype(float) + 1.0
        else:
            mv = np.asarray([float(M(v)) for v in yv])
            if np.any(np.diff(mv[np.argsort(yv)]) < 0):
                raise ValueError("M must be increasing in y.")

        def score(b):
            z = X @ b
            ind = (z[:, None] > z[None, :]).astype(float)
            np.fill_diagonal(ind, 0.0)
            return float(np.sum(mv[:, None] * ind)) / denom

    if d == 2:
        # The objective is PIECEWISE CONSTANT in the single free
        # coefficient t: with z_i = x1_i + t x2_i, the indicator
        # 1{z_i > z_j} flips exactly at t = -(x1_i - x1_j)/(x2_i -
        # x2_j). Evaluating on a grid costs O(n^2) PER grid point;
        # sorting the n^2 breakpoints and accumulating costs
        # O(n^2 log n) once. At n = 300 that is the difference
        # between minutes and milliseconds, and the result is exact
        # on the grid rather than approximated.
        a = X[:, 0][:, None] - X[:, 0][None, :]
        c = X[:, 1][:, None] - X[:, 1][None, :]
        wm = gt.copy() if variant == "mrc" else np.tile(mv[:, None], (1, n)).astype(float)
        np.fill_diagonal(wm, 0.0)
        flat_a, flat_c, flat_w = a.ravel(), c.ravel(), wm.ravel()
        keep = flat_w != 0.0
        flat_a, flat_c, flat_w = flat_a[keep], flat_c[keep], flat_w[keep]
        # value as t -> -inf: a + t c -> +inf exactly when c < 0
        base = float(flat_w[(flat_c < 0) | ((flat_c == 0) & (flat_a > 0))].sum())
        moving = flat_c != 0.0
        thr = -flat_a[moving] / flat_c[moving]
        # crossing upward (c > 0) switches the indicator ON, downward OFF
        sign = np.where(flat_c[moving] > 0, 1.0, -1.0) * flat_w[moving]
        order = np.argsort(thr)
        thr_s = thr[order]
        cum = base + np.concatenate([[0.0], np.cumsum(sign[order])])
        grid = np.linspace(-GRID_SCAN_HALF_WIDTH, GRID_SCAN_HALF_WIDTH,
                           GRID_SCAN_POINTS)
        vals = cum[np.searchsorted(thr_s, grid, side="right")] / denom
        k = int(np.argmax(vals))
        beta = np.array([1.0, float(grid[k])])
        negval = -float(vals[k])
    else:
        beta, negval = optimize_scale_normalized(lambda b: -score(b), d,
                                                 n_restarts=n_restarts,
                                                 seed=seed)
    return RichResult(payload={
        "beta": beta, "objective": -negval, "variant": variant,
        "requires_bandwidth": False,
        "asymptotically_efficient": False,
        "rate_exponent": -0.5, "inference": "bootstrap", "se": None,
        "n": int(n), "d": int(d),
        "method": "Rank correlation over orderings; no bandwidth, but not efficient and no analytic SE"})


def cheatsheet():
    return "hrzrank: no bandwidth at all -- but not efficient, and inference needs the bootstrap"
