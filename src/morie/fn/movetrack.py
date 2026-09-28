# morie.fn -- function file (rootcoder007/morie)
"""Animal-movement and spatial-network analysis: step decomposition of tracks, movement networks,
core-periphery structure, network robustness to node removal, the gamma/von Mises movement HMM, and a
Kalman-filtered correlated random walk (state-space movement model)."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._s03core import digamma

__all__ = [
    "track_steps",
    "movement_network",
    "core_periphery",
    "network_robustness",
    "movement_hmm",
    "crw_kalman",
]


def track_steps(x, y, t=None) -> RichResult:
    r"""Step-by-step decomposition of a trajectory, as ``adehabitatLT::as.ltraj``.

    For successive relocations: increments ``dx``, ``dy``, step length
    ``dist``, time step ``dt``, squared net displacement ``R2n`` from the
    start, absolute angle ``atan2(dy, dx)`` and relative (turning) angle, the
    change in absolute angle wrapped to ``(-pi, pi]``; undefined entries are ``None``.

    References
    ----------
    Calenge, C., Dray, S. and Royer-Carenzi, M. (2009). The concept of
    animals' trajectories from a data analysis perspective. *Ecological
    Informatics*, 4, 34-41.
    Turchin, P. (1998). *Quantitative Analysis of Movement*. Sinauer.

    Examples
    --------
    >>> r = track_steps([0, 1, 2, 2, 3], [0, 0, 1, 2, 2])
    >>> [round(v, 6) for v in r.dist[:4]], r.R2n
    ([1.0, 1.414214, 1.0, 1.0], [0.0, 1.0, 5.0, 8.0, 13.0])
    """
    X, Y = [float(v) for v in x], [float(v) for v in y]
    n = len(X)
    T = list(range(n)) if t is None else [float(v) for v in t]
    dx = [X[i + 1] - X[i] for i in range(n - 1)] + [None]
    dy = [Y[i + 1] - Y[i] for i in range(n - 1)] + [None]
    dist = [math.hypot(dx[i], dy[i]) for i in range(n - 1)] + [None]
    dt = [T[i + 1] - T[i] for i in range(n - 1)] + [None]
    r2 = [(X[i] - X[0]) ** 2 + (Y[i] - Y[0]) ** 2 for i in range(n)]
    ab = [math.atan2(dy[i], dx[i]) if dist[i] > 0 else None for i in range(n - 1)] + [None]
    rel = [None]
    for i in range(1, n - 1):
        if ab[i] is None or ab[i - 1] is None:
            rel.append(None)
            continue
        d = ab[i] - ab[i - 1]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        if d == -math.pi:
            d = math.pi
        rel.append(d)
    rel.append(None)
    return RichResult(
        payload={"dx": dx, "dy": dy, "dist": dist, "dt": dt, "R2n": r2, "abs_angle": ab, "rel_angle": rel}
    )


def movement_network(x, y, resolution: float, *, origin=(0.0, 0.0)) -> RichResult:
    r"""Movement network of a trajectory (Bastille-Rousseau et al. 2018): grid cells as nodes, transitions as edges.

    Relocations are binned into square cells of side ``resolution``; each move
    between distinct consecutive cells adds weight 1 to a directed edge.
    Returns the nodes (cell indices), the weighted edge list and per-node
    in/out degree, strength and visit counts.

    References
    ----------
    Bastille-Rousseau, G., Douglas-Hamilton, I., Blake, S., Northrup, J. M.
    and Wittemyer, G. (2018). Applying network theory to animal movements to
    identify properties of landscape space use. *Ecological Applications*, 28, 854-864.

    Examples
    --------
    >>> r = movement_network([0.5, 1.5, 1.6, 0.4, 1.5], [0.5, 0.5, 0.4, 0.6, 0.5], 1.0)
    >>> r.nodes, r.edges
    ([(0, 0), (1, 0)], [(0, 1, 2), (1, 0, 1)])
    """
    cells = [
        (int(math.floor((a - origin[0]) / resolution)), int(math.floor((b - origin[1]) / resolution)))
        for a, b in zip(x, y)
    ]
    nodes = []
    for c in cells:
        if c not in nodes:
            nodes.append(c)
    idx = {c: k for k, c in enumerate(nodes)}
    w = {}
    for a, b in zip(cells[:-1], cells[1:]):
        if a != b:
            key = (idx[a], idx[b])
            w[key] = w.get(key, 0) + 1
    edges = [(i, j, c) for (i, j), c in sorted(w.items())]
    m = len(nodes)
    return RichResult(
        payload={
            "nodes": nodes,
            "edges": edges,
            "out_degree": [sum(1 for i, _, _ in edges if i == k) for k in range(m)],
            "in_degree": [sum(1 for _, j, _ in edges if j == k) for k in range(m)],
            "strength": [sum(c for i, j, c in edges if k in (i, j)) for k in range(m)],
            "visits": [cells.count(c) for c in nodes],
        }
    )


def core_periphery(A, *, tol: float = 1e-12, maxit: int = 10000) -> RichResult:
    r"""Continuous core-periphery structure by the MINRES fit ``min sum_{i != j} (a_ij - c_i c_j)^2``.

    Damped fixed-point iteration of the stationarity condition
    ``c_i = sum_{j != i} a_ij c_j / sum_{j != i} c_j^2`` from the row sums
    (Boyd, Fitzgerald, Mahutga and Smith 2010; each step averages the old
    and new vectors, since the plain map can oscillate in scale); coreness
    is ``c`` (non-negative for non-negative ``A``) and the fit is the Pearson
    correlation of the off-diagonal ``a_ij`` with ``c_i c_j`` (Borgatti and Everett 1999).

    References
    ----------
    Borgatti, S. P. and Everett, M. G. (1999). Models of core/periphery
    structures. *Social Networks*, 21, 375-395.
    Boyd, J. P., Fitzgerald, W. J., Mahutga, M. C. and Smith, D. A. (2010).
    Computing continuous core/periphery structures for social relations data
    with MINRES/SVD. *Social Networks*, 32, 125-137.

    Examples
    --------
    >>> A = [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]]
    >>> [round(v, 4) for v in core_periphery(A).coreness]
    [1.4092, 0.7854, 0.7854, 0.4377]
    """
    M = [[float(v) for v in r] for r in A]
    n = len(M)
    c = [ssum(M[i][j] for j in range(n) if j != i) for i in range(n)]
    s = math.sqrt(ssum(v * v for v in c)) or 1.0
    c = [v / s for v in c]
    it = 0
    for it in range(1, maxit + 1):  # noqa: B007 (reported)
        sq = ssum(v * v for v in c)
        new = [ssum(M[i][j] * c[j] for j in range(n) if j != i) / (sq - c[i] * c[i]) for i in range(n)]
        new = [0.5 * (a + b) for a, b in zip(new, c)]  # damping: the plain map oscillates in scale
        diff = max(abs(a - b) for a, b in zip(new, c))
        c = new
        if diff < tol:
            break
    a = [M[i][j] for i in range(n) for j in range(n) if i != j]
    b = [c[i] * c[j] for i in range(n) for j in range(n) if i != j]
    ma, mb = ssum(a) / len(a), ssum(b) / len(b)
    sab = ssum((u - ma) * (v - mb) for u, v in zip(a, b))
    fit = sab / math.sqrt(ssum((u - ma) ** 2 for u in a) * ssum((v - mb) ** 2 for v in b))
    return RichResult(payload={"coreness": c, "fit": fit, "iterations": it})


def _largest_component(adj, alive):
    seen, best = set(), 0
    for s in range(len(adj)):
        if not alive[s] or s in seen:
            continue
        stack, size = [s], 0
        seen.add(s)
        while stack:
            v = stack.pop()
            size += 1
            for w in adj[v]:
                if alive[w] and w not in seen:
                    seen.add(w)
                    stack.append(w)
        best = max(best, size)
    return best


def network_robustness(A, *, strategy: str = "degree", seed: int = 1) -> RichResult:
    r"""Robustness of a network to node removal (Albert, Jeong and Barabasi 2000; Schneider et al. 2011).

    Nodes are removed one at a time, by highest current degree (adaptive
    targeted attack, ties to the lower index) or in a random Philox order
    (failures); ``s(Q)`` is the fraction of the ``N`` nodes in the largest
    connected component after ``Q`` removals and the robustness is
    ``R = (1/N) sum_{Q=1}^{N} s(Q)`` (about ``1/N`` for a large star under attack,
    about 1/2 for a complete graph). Resilience to failure uses ``strategy="random"``.

    References
    ----------
    Albert, R., Jeong, H. and Barabasi, A.-L. (2000). Error and attack
    tolerance of complex networks. *Nature*, 406, 378-382.
    Schneider, C. M., Moreira, A. A., Andrade, J. S., Havlin, S. and Herrmann,
    H. J. (2011). Mitigation of malicious attacks on networks. *PNAS*, 108, 3838-3841.

    Examples
    --------
    >>> star = [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    >>> network_robustness(star).R
    0.1875
    """
    M = [[float(v) for v in r] for r in A]
    n = len(M)
    adj = [[j for j in range(n) if M[i][j] != 0 and j != i] for i in range(n)]
    alive = [True] * n
    if strategy == "random":
        u = [float(v) for v in random_uniform(n, seed=seed)]
        order = sorted(range(n), key=lambda i: (u[i], i))
    elif strategy != "degree":
        raise ValueError("strategy must be 'degree' or 'random'")
    curve = []
    for q in range(n):
        if strategy == "degree":
            k = max((i for i in range(n) if alive[i]), key=lambda i: (sum(1 for j in adj[i] if alive[j]), -i))
        else:
            k = order[q]
        alive[k] = False
        curve.append(_largest_component(adj, alive) / n)
    return RichResult(payload={"R": ssum(curve) / n, "curve": curve})


def _trigamma(x):
    r = 0.0
    while x < 6.0:
        r += 1.0 / (x * x)
        x += 1.0
    f = 1.0 / (x * x)
    return r + 1.0 / x + f / 2.0 + f / x * (1.0 / 6.0 - f * (1.0 / 30.0 - f * (1.0 / 42.0 - f / 30.0)))


def _bessel_ratio(k):
    """A1(kappa) = I1(kappa) / I0(kappa) by the power series (kappa <= 500)."""
    h = k / 2.0
    t0 = t1 = 1.0
    s0, s1 = 1.0, 1.0
    for m in range(1, 2000):
        t0 *= h * h / (m * m)
        t1 *= h * h / (m * (m + 1))
        s0 += t0
        s1 += t1
        if t0 < 1e-17 * s0 and t1 < 1e-17 * s1:
            break
    return h * s1 / s0


def _vm_kappa(Rbar):
    if Rbar <= 1e-12:
        return 0.0
    k = Rbar * (2 - Rbar * Rbar) / (1 - Rbar * Rbar) if Rbar < 0.999 else 500.0
    k = min(k, 500.0)
    for _ in range(100):
        a = _bessel_ratio(k)
        da = 1 - a / k - a * a if k > 0 else 0.5
        step = (a - Rbar) / da
        k = min(max(k - step, 1e-8), 500.0)
        if abs(step) < 1e-13 * max(1.0, k):
            break
    return k


def _log_i0(k):
    h, t, s = k / 2.0, 1.0, 1.0
    for m in range(1, 2000):
        t *= h * h / (m * m)
        s += t
        if t < 1e-17 * s:
            break
    return math.log(s)


def movement_hmm(
    step, angle, n_states: int = 2, *, shape=None, scale=None, mu=None, kappa=None, maxit: int = 200, tol: float = 1e-10
) -> RichResult:
    r"""Hidden Markov model of animal movement with gamma step lengths and von Mises turning angles, fitted by EM.

    State ``k`` emits step lengths ``Gamma(shape_k, scale_k)`` and turning
    angles ``vonMises(mu_k, kappa_k)`` (missing angles, ``None``, contribute
    1); the Baum-Welch E-step uses scaled forward-backward recursions, the
    M-step weighted maximum likelihood (gamma shape by Newton on
    ``log k - psi(k) = log mean - mean log``, von Mises ``mu`` circular mean
    and ``kappa = A1^{-1}(Rbar)``). Returns the parameters, transition
    matrix, log-likelihood path and the Viterbi state sequence.

    References
    ----------
    Morales, J. M., Haydon, D. T., Frair, J., Holsinger, K. E. and Fryxell,
    J. M. (2004). Extracting more out of relocation data: building movement
    models as mixtures of random walks. *Ecology*, 85, 2436-2445.
    Michelot, T., Langrock, R. and Patterson, T. A. (2016). moveHMM: an R
    package for the statistical modelling of animal movement data using
    hidden Markov models. *Methods in Ecology and Evolution*, 7, 1308-1315.

    Examples
    --------
    >>> st = [0.2, 0.3, 0.25, 0.2, 2.5, 3.0, 2.8, 2.6, 0.3, 0.2, 0.25, 3.1, 2.9]
    >>> an = [None, 2.5, -2.8, 3.0, 0.1, -0.1, 0.05, 0.0, 2.9, -3.0, 2.7, 0.1, -0.05]
    >>> movement_hmm(st, an, shape=[2, 2], scale=[0.1, 1.5], mu=[3.1, 0], kappa=[1, 1]).states[:6]
    [0, 0, 0, 0, 1, 1]
    """
    S = [float(v) for v in step]
    Aang = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in angle]
    T, K = len(S), n_states
    sh = [float(v) for v in shape] if shape is not None else [2.0] * K
    sc = [float(v) for v in scale] if scale is not None else [(k + 1) * ssum(S) / T / K for k in range(K)]
    mu_ = [float(v) for v in mu] if mu is not None else [math.pi if k == 0 else 0.0 for k in range(K)]
    ka = [float(v) for v in kappa] if kappa is not None else [1.0] * K
    G = [[0.9 if i == j else 0.1 / (K - 1) for j in range(K)] for i in range(K)]
    delta = [1.0 / K] * K

    def emis():
        out = []
        for t in range(T):
            row = []
            for k in range(K):
                lp = (sh[k] - 1) * math.log(S[t]) - S[t] / sc[k] - math.lgamma(sh[k]) - sh[k] * math.log(sc[k])
                if Aang[t] is not None:
                    lp += ka[k] * math.cos(Aang[t] - mu_[k]) - math.log(2 * math.pi) - _log_i0(ka[k])
                row.append(math.exp(lp))
            out.append(row)
        return out

    lls = []
    for _ in range(maxit):
        P = emis()
        al, cs = [], []
        a = [delta[k] * P[0][k] for k in range(K)]
        for t in range(T):
            if t > 0:
                a = [ssum(al[t - 1][i] * G[i][k] for i in range(K)) * P[t][k] for k in range(K)]
            c = ssum(a)
            a = [v / c for v in a]
            al.append(a)
            cs.append(c)
        be = [[1.0] * K for _ in range(T)]
        for t in range(T - 2, -1, -1):
            be[t] = [ssum(G[k][j] * P[t + 1][j] * be[t + 1][j] for j in range(K)) / cs[t + 1] for k in range(K)]
        gam = [[al[t][k] * be[t][k] for k in range(K)] for t in range(T)]
        ll = ssum(math.log(c) for c in cs)
        lls.append(ll)
        xi = [[0.0] * K for _ in range(K)]
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[i][j] += al[t][i] * G[i][j] * P[t + 1][j] * be[t + 1][j] / cs[t + 1]
        G = [[xi[i][j] / ssum(xi[i]) for j in range(K)] for i in range(K)]
        delta = list(gam[0])
        for k in range(K):
            w = [gam[t][k] for t in range(T)]
            sw = ssum(w)
            m = ssum(w[t] * S[t] for t in range(T)) / sw
            s = math.log(m) - ssum(w[t] * math.log(S[t]) for t in range(T)) / sw
            kk = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
            for _ in range(100):
                f = math.log(kk) - digamma(kk) - s
                step_ = f / (1 / kk - _trigamma(kk))
                kk = max(kk - step_, 1e-8)
                if abs(step_) < 1e-13 * kk:
                    break
            sh[k], sc[k] = kk, m / kk
            ts = [t for t in range(T) if Aang[t] is not None]
            C = ssum(w[t] * math.cos(Aang[t]) for t in ts)
            Sn = ssum(w[t] * math.sin(Aang[t]) for t in ts)
            mu_[k] = math.atan2(Sn, C)
            ka[k] = _vm_kappa(math.sqrt(C * C + Sn * Sn) / ssum(w[t] for t in ts))
        if len(lls) > 1 and abs(lls[-1] - lls[-2]) < tol * (abs(lls[-1]) + 1):
            break
    P = emis()
    V = [[math.log(delta[k] + 1e-300) + math.log(P[0][k] + 1e-300) for k in range(K)]]
    back = []
    for t in range(1, T):
        row, bk = [], []
        for k in range(K):
            j = max(range(K), key=lambda i: (V[-1][i] + math.log(G[i][k] + 1e-300), -i))
            row.append(V[-1][j] + math.log(G[j][k] + 1e-300) + math.log(P[t][k] + 1e-300))
            bk.append(j)
        V.append(row)
        back.append(bk)
    s_ = [max(range(K), key=lambda k: (V[-1][k], -k))]
    for t in range(T - 2, -1, -1):
        s_.append(back[t][s_[-1]])
    s_.reverse()
    return RichResult(
        payload={"shape": sh, "scale": sc, "mu": mu_, "kappa": ka, "transition": G, "loglik": lls, "states": s_}
    )


def _kf_axis(y, g, s2, t2):
    """Kalman filter log-likelihood and smoothed positions of one axis of the CRW."""
    n = len(y)
    x = [y[0], 0.0]
    P = [[t2 + 1e6, 0.0], [0.0, s2 / max(1 - g * g, 1e-12)]]
    ll = 0.0
    xs, Ps, xp_l, Pp_l = [], [], [], []
    for t in range(n):
        if t > 0:
            xp = [x[0] + g * x[1], g * x[1]]
            Pp = [
                [P[0][0] + 2 * g * P[0][1] + g * g * P[1][1] + s2, g * P[0][1] + g * g * P[1][1] + s2],
                [0.0, g * g * P[1][1] + s2],
            ]
            Pp[1][0] = Pp[0][1]
        else:
            xp, Pp = x, P
        v = y[t] - xp[0]
        F = Pp[0][0] + t2
        ll += -0.5 * (math.log(2 * math.pi * F) + v * v / F) if t > 0 else 0.0
        K0, K1 = Pp[0][0] / F, Pp[1][0] / F
        x = [xp[0] + K0 * v, xp[1] + K1 * v]
        P = [
            [Pp[0][0] - K0 * Pp[0][0], Pp[0][1] - K0 * Pp[0][1]],
            [Pp[1][0] - K1 * Pp[0][0], Pp[1][1] - K1 * Pp[0][1]],
        ]
        xs.append(x)
        Ps.append(P)
        xp_l.append(xp)
        Pp_l.append(Pp)
    sm = [None] * n
    sm[-1] = list(xs[-1])
    for t in range(n - 2, -1, -1):
        Pp = Pp_l[t + 1]
        det = Pp[0][0] * Pp[1][1] - Pp[0][1] * Pp[1][0]
        inv = [[Pp[1][1] / det, -Pp[0][1] / det], [-Pp[1][0] / det, Pp[0][0] / det]]
        Fm = [[1.0, g], [0.0, g]]
        PFt = [[ssum(Ps[t][i][k] * Fm[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
        J = [[ssum(PFt[i][k] * inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        d = [sm[t + 1][i] - xp_l[t + 1][i] for i in range(2)]
        sm[t] = [xs[t][i] + ssum(J[i][k] * d[k] for k in range(2)) for i in range(2)]
    return ll, [v[0] for v in sm]


def crw_kalman(x, y, *, gamma=None, sigma=None, tau=None, cycles: int = 20) -> RichResult:
    r"""State-space correlated random walk with measurement error, by Kalman filtering and smoothing.

    Per axis the true displacement follows ``d_t = gamma d_{t-1} + e_t``,
    ``e_t ~ N(0, sigma^2)``, positions ``x_t = x_{t-1} + d_t``, and the
    observations add ``N(0, tau^2)`` error (the DCRW of Jonsen et al. 2005
    without rotation). The exact Gaussian likelihood comes from the Kalman
    filter (first observation conditioned on); unspecified parameters are
    estimated by maximum likelihood with cyclic golden-section searches over
    ``gamma`` in ``(0, 0.999)`` and ``log sigma``, ``log tau``. Smoothed
    positions are from the Rauch-Tung-Striebel smoother.

    References
    ----------
    Jonsen, I. D., Mills Flemming, J. and Myers, R. A. (2005). Robust
    state-space modeling of animal movement data. *Ecology*, 86, 2874-2880.
    Johnson, D. S., London, J. M., Lea, M.-A. and Durban, J. W. (2008).
    Continuous-time correlated random walk model for animal telemetry data.
    *Ecology*, 89, 1208-1215.

    Examples
    --------
    >>> xs = [0.0, 1.1, 2.3, 3.2, 4.4, 5.3, 6.6, 7.4]
    >>> ys = [0.0, 0.2, 0.1, 0.5, 0.4, 0.9, 1.0, 1.3]
    >>> r = crw_kalman(xs, ys, gamma=0.8, sigma=0.3, tau=0.1)
    >>> round(r.loglik, 8) < 0, len(r.x_smooth)
    (True, 8)
    """
    X, Y = [float(v) for v in x], [float(v) for v in y]

    def ll(g, s, t):
        return _kf_axis(X, g, s * s, t * t)[0] + _kf_axis(Y, g, s * s, t * t)[0]

    free = {"g": gamma is None, "s": sigma is None, "t": tau is None}
    steps = [math.hypot(X[i + 1] - X[i], Y[i + 1] - Y[i]) for i in range(len(X) - 1)]
    sd = math.sqrt(ssum(v * v for v in steps) / max(len(steps), 1)) or 1.0
    g = 0.5 if gamma is None else float(gamma)
    ls = math.log(sd / 2) if sigma is None else math.log(float(sigma))
    lt = math.log(sd / 10) if tau is None else math.log(float(tau))

    def gold(f, lo, hi):
        gr = (math.sqrt(5) - 1) / 2
        a1, a2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        f1, f2 = f(a1), f(a2)
        for _ in range(200):
            if f1 >= f2:
                hi, a2, f2 = a2, a1, f1
                a1 = hi - gr * (hi - lo)
                f1 = f(a1)
            else:
                lo, a1, f1 = a1, a2, f2
                a2 = lo + gr * (hi - lo)
                f2 = f(a2)
            if hi - lo < 1e-10:
                break
        return 0.5 * (lo + hi)

    if any(free.values()):
        for _ in range(cycles):
            if free["g"]:
                g = gold(lambda v, ls=ls, lt=lt: ll(v, math.exp(ls), math.exp(lt)), 1e-6, 0.999)
            if free["s"]:
                ls = gold(lambda v, g=g, lt=lt: ll(g, math.exp(v), math.exp(lt)), ls - 5, ls + 5)
            if free["t"]:
                lt = gold(lambda v, g=g, ls=ls: ll(g, math.exp(ls), math.exp(v)), lt - 8, lt + 5)
    s, t = math.exp(ls), math.exp(lt)
    lx, sx = _kf_axis(X, g, s * s, t * t)
    ly, sy = _kf_axis(Y, g, s * s, t * t)
    return RichResult(payload={"gamma": g, "sigma": s, "tau": t, "loglik": lx + ly, "x_smooth": sx, "y_smooth": sy})


def cheatsheet() -> str:
    return (
        "track_steps / movement_network / core_periphery / network_robustness / movement_hmm / crw_kalman -> "
        "movement and spatial networks."
    )
