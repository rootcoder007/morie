# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel GMM estimator."""

import math

from ._qpcore import ssum
from ._richresult import RichResult
from .sppanel import _cols, _golden, _lag, _ols
from .sppfe import _stack


def _q0(v, N, T):
    m = [ssum(v[t * N + i] for t in range(T)) / T for i in range(N)]
    return [v[t * N + i] - m[i] for t in range(T) for i in range(N)]


def _q1(v, N, T):
    m = [ssum(v[t * N + i] for t in range(T)) / T for i in range(N)]
    return [m[i] for _ in range(T) for i in range(N)]


def sppgmm(y, X, W, time_id, unit_id):
    r"""Kapoor-Kelejian-Prucha (2007) GM estimator of the random-effects spatial error panel model.

    y = X beta + u, u = lambda (I_T kron W) u + e, e = (iota_T kron
    I_N) mu + nu. From pooled OLS residuals u, ub = W_NT u and ubb
    = W_NT ub, the three within-moment conditions (Q0 = (I_T - J_T/T)
    kron I_N, divisor N(T - 1))

    E e'Q0e = sigma_nu^2, E eb'Q0eb = sigma_nu^2 tr(W'W)/N, E eb'Q0e = 0

    with e = u - lambda ub, eb = ub - lambda ubb, are fitted by
    nonlinear least squares in (lambda, sigma_nu^2) (sigma_nu^2
    profiled out, lambda by golden-section search); sigma_1^2 = e'Q1e /
    N (Q1 = J_T/T kron I_N) gives sigma_mu^2 = (sigma_1^2 -
    sigma_nu^2) / T. beta is then the feasible GLS estimate on the
    spatially Cochrane-Orcutt transformed data y - lambda W_NT y scaled by
    Q0 / sigma_nu + Q1 / sigma_1 (the KKP "initial" GM estimator; an
    intercept is added).

    References
    ----------
    Kapoor, M., Kelejian, H. H. and Prucha, I. R. (2007). Panel data models
    with spatially correlated error components. *Journal of Econometrics*
    140, 97-130.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> r = sppgmm(y, X, W, tid, uid)
    >>> round(r["lambda"], 8), [round(b, 8) for b in r["coefficients"]]
    (-0.07506872, [1.33281944, 0.77213661])
    """
    yv, Xm, N, T = _stack(y, X, time_id, unit_id)
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    Xa = [[1.0] + r for r in Xm]
    _, u = _ols(Xa, yv)
    ub = _lag(Wm, u, N, T)
    ubb = _lag(Wm, ub, N, T)
    d = N * (T - 1.0)

    def ip(a, b):
        return ssum(p * q for p, q in zip(a, b))

    qu, qub, qubb = _q0(u, N, T), _q0(ub, N, T), _q0(ubb, N, T)
    trww = ssum(Wm[i][j] * Wm[i][j] for i in range(N) for j in range(N)) / N
    g = [ip(u, qu) / d, ip(ub, qub) / d, ip(ub, qu) / d]
    G = [
        [2 * ip(u, qub) / d, -ip(ub, qub) / d, 1.0],
        [2 * ip(ubb, qub) / d, -ip(ubb, qubb) / d, trww],
        [(ip(u, qubb) + ip(ub, qub)) / d, -ip(ub, qubb) / d, 0.0],
    ]

    def fit(lam):
        r0 = [g[k] - G[k][0] * lam - G[k][1] * lam * lam for k in range(3)]
        s2 = ssum(G[k][2] * r0[k] for k in range(3)) / ssum(G[k][2] ** 2 for k in range(3))
        return ssum((r0[k] - G[k][2] * s2) ** 2 for k in range(3)), s2

    lam = _golden(lambda v: -fit(v)[0], -0.99, 0.99)
    s2nu = fit(lam)[1]
    e = [a - lam * b for a, b in zip(u, ub)]
    s21 = ip(e, _q1(e, N, T)) / N
    ys = [a - lam * b for a, b in zip(yv, _lag(Wm, yv, N, T))]
    xs = _cols([[a - lam * b for a, b in zip(c, _lag(Wm, c, N, T))] for c in _cols(Xa)])
    snu, s1 = math.sqrt(s2nu), math.sqrt(s21)

    def om(v):
        return [a / snu + b / s1 for a, b in zip(_q0(v, N, T), _q1(v, N, T))]

    beta, _ = _ols(_cols([om(c) for c in _cols(xs)]), om(ys))
    return RichResult(
        payload={
            "lambda": lam,
            "coefficients": beta,
            "sigma2_nu": s2nu,
            "sigma2_1": s21,
            "sigma2_mu": (s21 - s2nu) / T,
            "moment_objective": fit(lam)[0],
        }
    )


sppgmm_fn = sppgmm


def cheatsheet() -> str:
    return "sppgmm(y, X, W, time_id, unit_id) -> Kapoor-Kelejian-Prucha (2007) GM random-effects spatial error panel."
