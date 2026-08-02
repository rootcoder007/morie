# morie.fn -- function file (rootcoder007/morie)
"""Ordered-response maximum-score estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_ordered_max_score"]


def horowitz_ordered_max_score(x, y, thresholds=None, smoothed=False, h=None,
                               n_restarts=8, seed=0):
    r"""Maximum-score estimator for an ordered-response model
    (Horowitz Sec. 4.4.3), equation (4.43):

    .. math:: S_{n,OR}(b) = \frac1n \sum_i \Big| W_i -
              \sum_{m=0}^{M-1} \mathbf 1\{X_i'b > \alpha_m\}\Big|,

    with :math:`Y = \mathbf 1\{\alpha_{m-1} < Y^* \le \alpha_m\}`,
    :math:`Y^* = X'\beta + U`, and
    :math:`W = \sum_{m=0}^{M-1}\mathbf 1\{Y^* > \alpha_m\}` observable
    from Y. Because :math:`\mathrm{median}(U|X) = 0` implies
    :math:`\mathrm{median}(W|X = x) = \sum_m \mathbf 1\{x'\beta >
    \alpha_m\}` (4.42), the estimator is a median regression.

    **This objective is MINIMISED.** The book prints "maximize" above
    (4.43) but the next sentence calls the result a median-regression
    estimator, and a median regression minimises absolute deviations.
    Measured on a simulated ordered model with
    :math:`\beta = (1, -0.7)` and n = 4000, minimising over a grid
    returns -0.75 while maximising runs to the boundary of the search
    region (-3, 3) and stops at 3.0. The printed "maximize" is
    therefore treated as an erratum, and ``sense`` records which way
    the objective was taken so the choice is visible rather than
    silent.

    Unlike the binary-response case this model needs NO scale
    normalisation -- levels of :math:`Y^*` are partly observable
    through the categories, whereas in a binary model only the sign
    is. The normalisation here is Lee's (1992): the first component
    of :math:`\beta` is fixed at 1 and :math:`\alpha_1 = 0`, which
    fixes location instead.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Covariates.
    y : array-like of integers in 0..M-1, shape (n,)
        Ordered response category.
    thresholds : array-like, optional
        The known cut-points :math:`\alpha_1, \dots, \alpha_{M-1}`.
        When omitted they are estimated jointly with beta under Lee's
        normalisation :math:`\alpha_1 = 0`.
    smoothed : bool, default False
        Use the Melenberg-van Soest smoothed objective
        :math:`S_{n,SOR}` instead, which is MAXIMISED (see below).
    h : float, optional
        Bandwidth for the smoothed form; ``n**(-1/5)`` otherwise.
    n_restarts : int, default 8
        Restarts, since (4.43) is a step function.
    seed : int, default 0
        RNG seed for the restarts.

    Returns
    -------
    RichResult
        keys: ``beta``, ``thresholds``, ``objective``, ``sense``
        ("minimised" or "maximised"), ``thresholds_estimated``,
        ``scale_normalisation_required`` (False), ``smoothed``,
        ``bandwidth``, ``M``, ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 4.4.3, eqs. (4.41)-(4.43) and
    Theorem 4.11; Kooreman and Melenberg (1989), Lee (1992),
    Melenberg and van Soest (1996).
    """
    from ._sci_core import optimize
    from . import _stats_core as stats

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    yi = np.asarray(yv, dtype=int)
    if not np.allclose(yi, np.asarray(yv, dtype=float)):
        raise ValueError("y must hold integer category labels.")
    if yi.min() < 0:
        raise ValueError("y categories must start at 0.")
    n, d = X.shape
    M = int(yi.max()) + 1
    if M < 3:
        raise ValueError(
            f"an ordered-response model needs at least 3 categories, got {M}.")
    if d < 2:
        raise ValueError(f"need at least 2 covariates, got {d}.")

    # W = sum_{m=0}^{M-1} I(Y* > alpha_m); alpha_0 = -inf contributes 1,
    # and Y = m means Y* passed exactly m of the finite cut-points
    W = 1.0 + yi.astype(float)

    known = thresholds is not None
    if known:
        a0 = np.asarray(thresholds, dtype=float).ravel()
        if a0.size != M - 1:
            raise ValueError(
                f"thresholds must have M-1 = {M - 1} entries, got {a0.size}.")
        if np.any(np.diff(a0) <= 0):
            raise ValueError("thresholds must be strictly increasing.")
    else:
        # Lee's normalisation: alpha_1 = 0, the rest free and ordered
        a0 = np.concatenate([[0.0], np.arange(1.0, M - 1)])

    hh = float(n ** (-0.2)) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")

    def unpack(z):
        b = np.r_[1.0, z[:d - 1]]
        if known:
            return b, a0
        # keep the cut-points ordered by cumulating positive gaps
        gaps = np.abs(z[d - 1:])
        return b, np.concatenate([[0.0], np.cumsum(gaps)]) if M > 2 else \
            np.array([0.0])

    def objective(z):
        b, a = unpack(z)
        v = X @ b
        if smoothed:
            # S_{n,SOR}(b, a) = (1/n) sum_i sum_m (2 W_im - 1)
            # K((X_i'b - a_m)/h): a genuine max-score objective, so it
            # is MAXIMISED and negated here
            Wim = (yi[:, None] > np.arange(M - 1)[None, :]).astype(float)
            s = np.sum((2.0 * Wim - 1.0) *
                       stats.norm.cdf((v[:, None] - a[None, :]) / hh))
            return -s / n
        pred = 1.0 + np.sum(v[:, None] > a[None, :], axis=1)
        return float(np.mean(np.abs(W - pred)))

    if known:
        # only beta is free, so the shared scale-normalised routine
        # applies -- and for d = 2 it scans the grid exhaustively
        # rather than trusting a simplex on a step function
        from ._horowitz import optimize_scale_normalized

        beta, val = optimize_scale_normalized(
            lambda b: objective(b[1:]), d, n_restarts=n_restarts, seed=seed)
        return RichResult(payload={
            "beta": beta, "thresholds": a0,
            "objective": -val if smoothed else val,
            "sense": "maximised" if smoothed else "minimised",
            "thresholds_estimated": False,
            "scale_normalisation_required": False,
            "smoothed": bool(smoothed),
            "bandwidth": hh if smoothed else None,
            "M": M, "n": int(n), "d": int(d),
            "method": "Ordered max score (4.43); a median regression, so absolute deviations are MINIMISED"})

    k = (d - 1) + max(M - 2, 0)
    rng = np.random.default_rng(seed)
    starts = [np.zeros(k)] + [rng.standard_normal(k) for _ in range(int(n_restarts))]
    best, best_val = None, np.inf
    for st in starts:
        r = optimize.minimize(objective, st, method="Nelder-Mead",
                              options={"maxiter": 5000, "fatol": 1e-9})
        if r.fun < best_val:
            best_val, best = float(r.fun), r.x
    beta, alpha = unpack(best)
    return RichResult(payload={
        "beta": beta, "thresholds": alpha,
        "objective": -best_val if smoothed else best_val,
        "sense": "maximised" if smoothed else "minimised",
        "thresholds_estimated": not known,
        "scale_normalisation_required": False,
        "smoothed": bool(smoothed),
        "bandwidth": hh if smoothed else None,
        "M": M, "n": int(n), "d": int(d),
        "method": "Ordered max score (4.43); a median regression, so absolute deviations are MINIMISED"})


def cheatsheet():
    return "hrzormsc: (4.43) prints 'maximize' but the estimator is a median regression -- minimise"
