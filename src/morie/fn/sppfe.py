# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel fixed effects (within) estimator."""

from .sppanel import spatial_panel_ml


def _flat(v):
    return [float(a) for a in (v.tolist() if hasattr(v, "tolist") else v)]


def _rows(A):
    rows = A.tolist() if hasattr(A, "tolist") else A
    return [[float(v) for v in (r if hasattr(r, "__len__") else [r])] for r in rows]


def _stack(y, X, time_id, unit_id):
    """Rows sorted by (period, unit): the first N rows are period 1 with units in increasing id order.

    Constant columns of X are dropped (the effects or the pooled intercept take their place).
    Returns (y, X rows, N, T).
    """
    yv = _flat(y)
    Xm = _rows(X) if X is not None else [[] for _ in yv]
    t = list(time_id.tolist() if hasattr(time_id, "tolist") else time_id)
    u = list(unit_id.tolist() if hasattr(unit_id, "tolist") else unit_id)
    order = sorted(range(len(yv)), key=lambda k: (t[k], u[k]))
    N, T = len(set(u)), len(set(t))
    if len(yv) != N * T:
        raise ValueError("the panel must be balanced: every unit observed in every period")
    keep = [c for c in range(len(Xm[0])) if len({r[c] for r in Xm}) > 1]
    return [yv[k] for k in order], [[Xm[k][c] for c in keep] for k in order], N, T


def sppfe(y, X, W, time_id, unit_id, lee_yu=False):
    r"""Fixed-effects (within) spatial lag panel model by maximum likelihood (Elhorst 2003).

    y_it = rho sum_j w_ij y_jt + x_it' beta + mu_i + e_it: the unit effects
    are removed by the within transformation and rho maximises the
    concentrated log-likelihood -NT/2 log SSE(rho) + T log|I - rho W|.
    Rows are matched to W by sorting on (period, unit id); constant
    columns of X are dropped. lee_yu applies the Lee-Yu (2010)
    variance correction T/(T - 1). Thin front-end to
    :func:`morie.fn.sppanel.spatial_panel_ml` (as splm::spml(model =
    "within", lag = TRUE)).

    References
    ----------
    Elhorst, J. P. (2003). Specification and estimation of spatial panel data
    models. *International Regional Science Review* 26, 244-268.
    Lee, L.-F. and Yu, J. (2010). Estimation of spatial autoregressive panel
    data models with fixed effects. *Journal of Econometrics* 154, 165-185.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> round(sppfe(y, X, W, tid, uid)["rho"], 8)
    -0.0808892
    """
    yv, Xm, N, _ = _stack(y, X, time_id, unit_id)
    return spatial_panel_ml(yv, Xm, W, N, model="lag", effects="individual", lee_yu=lee_yu)


sppfe_fn = sppfe


def cheatsheet() -> str:
    return "sppfe(y, X, W, time_id, unit_id) -> within (FE) spatial lag panel by ML (Elhorst 2003)."
