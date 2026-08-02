# morie.fn -- function file (rootcoder007/morie)
"""Maximum-score estimator for panel data with fixed effects."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_panel_max_score"]


def horowitz_panel_max_score(x, y, n_periods, smoothed=True, h=None,
                             n_restarts=8, seed=0):
    r"""Maximum-score estimator for panel binary response with fixed
    effects (Horowitz Sec. 4.4.2), equations (4.39) and (4.40):

    .. math:: S_{ms,pan}(b) = \frac1n \sum_i \sum_{t=2}^{T}
              \sum_{r<t} (Y_{it} - Y_{ir})\,
              \mathbf 1\{W_{itr}'b \ge 0\},
              \qquad W_{itr} = X_{it} - X_{ir},

    maximised subject to :math:`|b_1| = 1`; the smoothed form (4.40)
    replaces the indicator with :math:`K(W_{itr}'b/h_n)`. For
    :math:`T = 2` this is exactly (4.39).

    The model is :math:`Y_{it} = \mathbf 1\{X_{it}'\beta + U_i +
    \varepsilon_{it} > 0\}` with :math:`U_i` an unrestricted
    individual effect. Differencing removes :math:`U_i` entirely: no
    distributional assumption on it is needed and it may be
    correlated with the regressors, exactly as in a linear
    fixed-effects model. The same differencing costs the same things
    a linear fixed-effects model gives up:

    * an INTERCEPT is not identified;
    * any regressor CONSTANT within an individual is not identified,
      because its difference is identically zero;
    * only pairs with :math:`Y_{it} \ne Y_{ir}` contribute, since
      identical responses give a zero weight.

    Those are reported rather than assumed away: constant columns are
    detected and named in ``unidentified_columns``.

    Parameters
    ----------
    x : array-like, shape (n, T, d) or (n * T, d)
        Covariates. A 2-D array is reshaped using ``n_periods``, with
        the T observations of an individual CONTIGUOUS.
    y : array-like of {0, 1}, shape (n, T) or (n * T,)
        Binary responses, laid out to match ``x``.
    n_periods : int
        Number of periods T (at least 2).
    smoothed : bool, default True
        Use (4.40) rather than the discontinuous (4.39).
    h : float, optional
        Bandwidth for the smoothed form; ``n**(-1/5)`` otherwise.
    n_restarts : int, default 8
        Restarts, since (4.39) is a step function of b.
    seed : int, default 0
        RNG seed for the restarts.

    Returns
    -------
    RichResult
        keys: ``beta``, ``score``, ``n_pairs``,
        ``n_discordant_pairs`` (those actually contributing),
        ``unidentified_columns``, ``intercept_identified`` (False),
        ``smoothed``, ``bandwidth``, ``rate_exponent``, ``n``, ``T``,
        ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 4.4.2, eqs. (4.37)-(4.40),
    Theorems 4.9-4.10; Manski (1987), Charlier et al. (1995).
    """
    from scipy import stats

    from ._horowitz import optimize_scale_normalized

    T = int(n_periods)
    if T < 2:
        raise ValueError(f"need at least 2 periods, got {T}.")
    X = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if X.ndim == 2:
        if X.shape[0] % T:
            raise ValueError(
                f"x has {X.shape[0]} rows, not a multiple of n_periods={T}.")
        X = X.reshape(X.shape[0] // T, T, X.shape[1])
    elif X.ndim != 3:
        raise ValueError("x must be (n, T, d) or (n*T, d).")
    if X.shape[1] != T:
        raise ValueError(f"x has {X.shape[1]} periods, expected {T}.")
    yv = yv.reshape(-1, T) if yv.ndim == 1 else yv
    if yv.shape != X.shape[:2]:
        raise ValueError(f"y has shape {yv.shape}, expected {X.shape[:2]}.")
    if not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError("y must be binary 0/1.")
    n, _, d = X.shape
    if d < 2:
        raise ValueError(f"need at least 2 covariates, got {d}.")

    # every ordered pair r < t, stacked
    dw, dy = [], []
    for t in range(1, T):
        for r in range(t):
            dw.append(X[:, t, :] - X[:, r, :])
            dy.append(yv[:, t] - yv[:, r])
    W = np.vstack(dw)
    dY = np.concatenate(dy)

    # a covariate constant within every individual differences to zero
    # and carries no information about beta -- say so instead of
    # returning a number for it
    const_cols = [int(j) for j in range(d)
                  if np.allclose(X[:, :, j] - X[:, :1, j], 0.0)]

    hh = float(n ** (-0.2)) if h is None else float(h)
    if hh <= 0:
        raise ValueError(f"bandwidth must be positive, got {hh}.")

    def score(b):
        v = W @ b
        ind = stats.norm.cdf(v / hh) if smoothed else (v >= 0.0).astype(float)
        return float(np.sum(dY * ind)) / n

    beta, negval = optimize_scale_normalized(lambda b: -score(b), d,
                                             n_restarts=n_restarts, seed=seed)
    return RichResult(payload={
        "beta": beta, "score": -negval,
        "n_pairs": int(dY.size),
        "n_discordant_pairs": int(np.sum(dY != 0.0)),
        "unidentified_columns": const_cols,
        "intercept_identified": False,
        "smoothed": bool(smoothed),
        "bandwidth": hh if smoothed else None,
        "rate_exponent": -0.4 if smoothed else -1.0 / 3.0,
        "n": int(n), "T": T, "d": int(d),
        "method": "Panel max score (4.39)/(4.40); differencing kills U_i and the intercept with it"})


def cheatsheet():
    return "hrzpanms: differencing removes U_i -- and the intercept and time-constant regressors too"
