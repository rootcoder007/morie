# morie.fn -- function file (rootcoder007/morie)
"""Conditional prediction of Y given X after transformation model estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_conditional_prediction"]


def horowitz_conditional_prediction(x, y_threshold, T_hat, F_hat, beta_hat,
                                    gamma=None, y_grid=None, u_grid=None):
    r"""Prediction from a fitted transformation model (Horowitz
    Sec. 6.4), built on

    .. math:: P(Y \le y \mid X = x) = F\big[T(y) - x'\beta\big],

    which is just (6.1) rearranged, and on the gamma-quantile
    predictor

    .. math:: y_\gamma(x) = T^{-1}(x'\beta + u_\gamma),
              \qquad
              y_{n\gamma}(x) = \inf\{y : T_n(y) > x'b_n + u_{n\gamma}\},

    with :math:`u_{n\gamma} = \inf\{u : F_n(u) \ge \gamma\}`.

    The section's actual result is a NEGATIVE one, and it is the
    reason a quantile appears at all: when T is nonparametric,
    :math:`E(Y|X = x)` CANNOT be estimated at rate
    :math:`n^{-1/2}`. The root-n estimator
    :math:`n^{-1}\sum_i T_n^{-1}(U_{ni} + x'b_n)` requires T known up
    to a finite-dimensional parameter, because it needs T estimated
    at root-n accuracy over the WHOLE support of Y -- and Sec. 6.3.2
    only delivers that on a compact interval strictly inside the
    support. A conditional median or other quantile, by contrast, is
    usually root-n estimable, because it only needs :math:`F_n`
    accurate in a neighbourhood of :math:`u_\gamma`.

    So the honest output here is a probability and a quantile;
    ``mean_root_n_estimable`` is returned as False to keep that
    limitation attached to the result rather than left in the prose.

    Parameters
    ----------
    x : array-like, shape (d,) or (m, d)
        Covariate values to predict at.
    y_threshold : array-like
        The y at which :math:`P(Y \le y|X = x)` is wanted.
    T_hat : array-like or callable
        :math:`T_n` on ``y_grid``, or a callable of y.
    F_hat : array-like or callable
        :math:`F_n` on ``u_grid``, or a callable of u.
    beta_hat : array-like, shape (d,)
        Coefficients, rescaled to :math:`|b_1| = 1`.
    gamma : float in (0, 1), optional
        Quantile level for the predictor; the median otherwise.
    y_grid, u_grid : array-like, optional
        Required when ``T_hat`` / ``F_hat`` are arrays.

    Returns
    -------
    RichResult
        keys: ``probability``, ``quantile``, ``gamma``, ``u_gamma``,
        ``index`` (:math:`x'b`), ``mean_root_n_estimable`` (False),
        ``quantile_root_n_estimable`` (True), ``n_points``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 6.4 (predicting Y conditional on
    X); Cheng et al. (1997), Horowitz (1996).
    """
    from ._hrz_transform import normalize_scale

    b = normalize_scale(beta_hat)
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[1] != b.size:
        X = X.T
    if X.shape[1] != b.size:
        raise ValueError(
            f"x must have {b.size} columns to match beta_hat.")
    idx = X @ b
    yq = np.atleast_1d(np.asarray(y_threshold, dtype=float)).ravel()

    if callable(T_hat):
        T_of = T_hat
        if y_grid is None:
            raise ValueError("y_grid is required to invert a callable T_hat.")
        yg = np.atleast_1d(np.asarray(y_grid, dtype=float))
        Tg = np.array([float(T_of(v)) for v in yg])
    else:
        if y_grid is None:
            raise ValueError("y_grid is required when T_hat is an array.")
        yg = np.atleast_1d(np.asarray(y_grid, dtype=float))
        Tg = np.atleast_1d(np.asarray(T_hat, dtype=float))
        if Tg.size != yg.size:
            raise ValueError(
                f"T_hat has {Tg.size} entries for {yg.size} grid points.")
        def T_of(v):
            return float(np.interp(v, yg, Tg))
    if np.any(np.diff(Tg) < 0):
        raise ValueError("T_hat must be non-decreasing; assumption HT4 makes "
                         "T strictly increasing.")

    if callable(F_hat):
        F_of = F_hat
        if u_grid is None:
            raise ValueError("u_grid is required to invert a callable F_hat.")
        ug = np.atleast_1d(np.asarray(u_grid, dtype=float))
        Fg = np.array([float(F_of(v)) for v in ug])
    else:
        if u_grid is None:
            raise ValueError("u_grid is required when F_hat is an array.")
        ug = np.atleast_1d(np.asarray(u_grid, dtype=float))
        Fg = np.atleast_1d(np.asarray(F_hat, dtype=float))
        if Fg.size != ug.size:
            raise ValueError(
                f"F_hat has {Fg.size} entries for {ug.size} grid points.")
        def F_of(v):
            return float(np.interp(v, ug, Fg))
    if np.any(Fg < 0) or np.any(Fg > 1):
        raise ValueError("F_hat must lie in [0, 1].")

    # P(Y <= y | X = x) = F[T(y) - x'b]
    prob = np.array([[F_of(T_of(v) - z) for v in yq] for z in idx])

    g = 0.5 if gamma is None else float(gamma)
    if not 0.0 < g < 1.0:
        raise ValueError(f"gamma must lie in (0, 1), got {g}.")
    # u_ngamma = inf{u : F_n(u) >= gamma}
    hit = np.nonzero(Fg >= g)[0]
    u_g = float(ug[hit[0]]) if hit.size else float(ug[-1])
    # y_ngamma(x) = inf{y : T_n(y) > x'b + u_ngamma}
    quant = np.empty(idx.size)
    for i, z in enumerate(idx):
        over = np.nonzero(Tg > z + u_g)[0]
        quant[i] = float(yg[over[0]]) if over.size else np.nan

    if idx.size == 1:
        prob_out = float(prob[0, 0]) if yq.size == 1 else prob[0]
    else:
        prob_out = prob[:, 0] if yq.size == 1 else prob
    return RichResult(payload={
        "probability": prob_out,
        "quantile": quant if idx.size > 1 else float(quant[0]),
        "gamma": g, "u_gamma": u_g,
        "index": idx if idx.size > 1 else float(idx[0]),
        "mean_root_n_estimable": False,
        "quantile_root_n_estimable": True,
        "n_points": int(idx.size),
        "method": "P(Y<=y|x) = F[T(y) - x'b]; the conditional MEAN is not root-n estimable when T is nonparametric"})


def cheatsheet():
    return "hrzycp: with nonparametric T the conditional mean is NOT root-n -- predict a quantile"
