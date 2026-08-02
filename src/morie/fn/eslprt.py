# morie.fn -- function file (rootcoder007/morie)
"""Partial dependence -- Friedman (2001), ESL Sec 10.13.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["esl_partial_dependence"]


def esl_partial_dependence(model, X, S, grid=None, n_grid=20):
    r"""Partial dependence of ``model`` on the variables in ``S``.

    .. math::
        \bar f_S(x_S) = \frac{1}{n}\sum_{i=1}^{n} f\!\left(x_S,\; x_{iC}\right),

    the prediction averaged over the *observed* joint distribution of the
    complement :math:`x_C`, with :math:`x_S` held fixed. This is a marginal
    of the fitted surface, not a conditional expectation: it answers "what
    does the model do if I move :math:`x_S`", not "what is E[y | x_S]".

    The known hazard, which ESL states plainly: the average is taken over the
    marginal of :math:`x_C`, so when :math:`x_S` and :math:`x_C` are
    correlated the model is evaluated at combinations that never occur in the
    data, and the curve there is extrapolation. This is reported --
    ``extrapolation_warning`` flags grid points whose implied combinations
    are far from any observed row.

    Parameters
    ----------
    model : callable
        ``model(X) -> predictions``, taking an ``(m, p)`` array.
    X : array-like
        Data ``(n, p)`` supplying the complement distribution.
    S : int or sequence of int
        Column index/indices to vary.
    grid : array-like, optional
        Values of ``x_S`` to evaluate. Defaults to a quantile grid.
    n_grid : int
        Grid size when ``grid`` is not given.

    Returns
    -------
    RichResult
        ``grid``, ``pd`` (the curve), ``centered`` (mean-zero version),
        ``extrapolation_warning``.

    References
    ----------
    Friedman, J. H. (2001). Greedy function approximation: A gradient
        boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    For an additive model the partial-dependence curve reproduces the
    component exactly, up to the additive constant.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(-2, 2, (400, 2))
    >>> f = lambda Z: 3 * Z[:, 0] + Z[:, 1] ** 2
    >>> r = esl_partial_dependence(f, X, S=0, n_grid=9)
    >>> slope = np.polyfit(r["grid"].ravel(), r["pd"], 1)[0]
    >>> bool(abs(slope - 3.0) < 1e-8)
    True

    Centring makes the curve mean-zero, which is how it is normally plotted.

    >>> bool(abs(r["centered"].mean()) < 1e-9)
    True

    With correlated inputs the grid reaches combinations that never occur,
    and those points are flagged rather than plotted silently.

    >>> z = rng.normal(size=300)
    >>> Xc = np.column_stack([z, z + rng.normal(0, 0.05, 300)])   # near-collinear
    >>> rc = esl_partial_dependence(f, Xc, S=0, n_grid=9)
    >>> bool(rc["extrapolation_warning"].any())
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    S = np.atleast_1d(np.asarray(S, dtype=int))
    if np.any((S < 0) | (S >= p)):
        raise ValueError(f"S contains a column index outside 0..{p - 1}")
    if S.size != np.unique(S).size:
        raise ValueError("S must not repeat a column")

    if grid is None:
        qs = np.linspace(0.05, 0.95, int(n_grid))
        axes = [np.quantile(X[:, j], qs) for j in S]
        G = np.array(np.meshgrid(*axes, indexing="ij")).reshape(S.size, -1).T
    else:
        G = np.atleast_2d(np.asarray(grid, dtype=float))
        if G.shape[1] != S.size:
            G = G.reshape(-1, S.size)

    pd = np.empty(G.shape[0])
    warn = np.zeros(G.shape[0], dtype=bool)
    Dxx = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    np.fill_diagonal(Dxx, np.inf)
    ref_nn = float(np.median(Dxx.min(axis=1)))
    for t, gv in enumerate(G):
        Z = X.copy()
        Z[:, S] = gv
        pd[t] = float(np.mean(np.asarray(model(Z), dtype=float).ravel()))
        # Extrapolation lives in the JOINT, not in x_S: every grid value is a
        # quantile of x_S and so always near observed x_S values. What can be
        # unobserved is the pair (g, x_iC). Compare how far each synthetic row
        # sits from the real cloud against how far real rows sit from each
        # other.
        dz = np.sqrt(((Z[:, None] - X[None]) ** 2).sum(-1)).min(axis=1)
        warn[t] = bool(np.median(dz) > 2.0 * ref_nn)

    return RichResult(
        title="Partial dependence",
        summary_lines=[("n", n), ("variables", list(map(int, S))),
                       ("grid points", int(G.shape[0]))],
        payload={
            "grid": G, "pd": pd, "centered": pd - pd.mean(),
            "extrapolation_warning": warn,
            "S": S, "n": int(n),
            "method": "esl_partial_dependence",
        },
    )


def cheatsheet():
    return "eslprt: marginal of the FITTED surface, not E[y|x_S]; correlated inputs make it extrapolate (flagged)"
