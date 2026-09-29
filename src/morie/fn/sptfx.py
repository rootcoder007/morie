# morie.fn -- function file (rootcoder007/morie)
"""Spatial probit fixed-effects panel."""

from ._qpcore import ssum
from .spdiscrete import spatial_probit_gmm
from .spprmf import _rows


def _panel(y, X, W, unit_id, time_id):
    """Stack by (period, unit) and build I_T kron W; returns y, X rows (non-constant), big W, unit index."""
    yv = [float(v) for v in y]
    Xm = _rows(X)
    u = list(unit_id)
    t = list(time_id) if time_id is not None else None
    ids = sorted(set(u))
    N = len(ids)
    if t is None:
        seen = {}
        t = []
        for g in u:
            t.append(seen.get(g, 0))
            seen[g] = seen.get(g, 0) + 1
    order = sorted(range(len(yv)), key=lambda k: (t[k], ids.index(u[k])))
    T = len(yv) // N
    if len(yv) != N * T:
        raise ValueError("the panel must be balanced")
    keep = [c for c in range(len(Xm[0])) if len({r[c] for r in Xm}) > 1]
    Wm = _rows(W)
    big = [[Wm[i % N][j % N] if i // N == j // N else 0.0 for j in range(N * T)] for i in range(N * T)]
    unit = [ids.index(u[k]) for k in order]
    return [yv[k] for k in order], [[Xm[k][c] for c in keep] for k in order], big, unit, N


def sptfx(y, X, W, unit_id, time_id=None):
    r"""Fixed-effects spatial probit panel by GMM with unit dummies.

    y*_it = rho sum_j w_ij y*_jt + x_it' beta + alpha_i + e_it: the panel is
    stacked by period (time_id, or the order of appearance within each
    unit) with block weights I_T kron W, and the unit effects enter as
    dummy variables (the first unit is the baseline; with the intercept)
    in the Pinkse-Slade GMM spatial probit of
    :func:`morie.fn.spdiscrete.spatial_probit_gmm` (instruments [X, WX]).
    The dummy-variable probit carries the incidental-parameters bias of order
    1/T (Neyman and Scott 1948; Greene 2004). Coefficients: intercept,
    non-constant X columns, then the N - 1 unit effects.

    References
    ----------
    Pinkse, J. and Slade, M. E. (1998). Contracting in space: an application
    of spatial statistics to discrete-choice models. *Journal of
    Econometrics* 85, 125-154.
    Greene, W. (2004). The behaviour of the maximum likelihood estimator of
    limited dependent variable models in the presence of fixed effects.
    *Econometrics Journal* 7, 98-119.

    Examples
    --------
    >>> import math
    >>> W = [[0.5 if abs(i - j) in (1, 5) else 0.0 for j in range(6)] for i in range(6)]
    >>> uid = list(range(6)) * 10
    >>> X = [[math.sin(1.7 * k) + 0.3 * math.cos(0.4 * k)] for k in range(60)]
    >>> y = [1 if X[k][0] + 0.8 * math.sin(3.3 * k + 1) + 0.3 * ((k % 6) - 3) / 6 > 0 else 0 for k in range(60)]
    >>> round(sptfx(y, X, W, uid)["rho"], 8)
    0.25887525
    """
    yv, Xm, big, unit, N = _panel(y, X, W, unit_id, time_id)
    D = [[1.0] + r + [1.0 if unit[k] == g else 0.0 for g in range(1, N)] for k, r in enumerate(Xm)]
    # lagged unit dummies are combinations of the dummies, so only the covariates are lagged
    kx = len(Xm[0])
    lagx = [[ssum(big[i][j] * Xm[j][c] for j in range(len(Xm))) for c in range(kx)] for i in range(len(Xm))]
    return spatial_probit_gmm(yv, D, big, Z=[d + lx for d, lx in zip(D, lagx)])


sptfx_fn = sptfx


def cheatsheet() -> str:
    return "sptfx(y, X, W, unit_id) -> FE spatial probit panel: GMM spatial probit with unit dummies, I_T kron W."
