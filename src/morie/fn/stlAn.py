# SPDX-License-Identifier: AGPL-3.0-or-later
"""STL decomposition (Cleveland et al. 1990) with residual outliers."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["stlAn", "stl_anomaly", "stl_decompose"]


def _tricube(u):
    if u >= 1.0:
        return 0.0
    t = 1.0 - u * u * u
    return t * t * t


def _loess_at(xs, ys, x0, q, degree, rho=None):
    # loess fitted value at x0 (Cleveland et al. 1990, Sec. 2.1):
    # q nearest points, tricube neighbourhood weights
    # W(|x_i - x0| / lambda_q(x0)); lambda_q = distance of the q-th
    # nearest (times q/n when q > n); local polynomial of the given
    # degree fitted by weighted least squares; optional robustness
    # weights rho multiply the neighbourhood weights (Sec. 2.4).
    n = len(xs)
    if q >= n:
        lam = 0.0
        for xi in xs:
            d = abs(xi - x0)
            if d > lam:
                lam = d
        lam *= q / float(n)
        idx = list(range(n))
    else:
        order = sorted(range(n), key=lambda i: (abs(xs[i] - x0), xs[i]))
        idx = order[:q]
        lam = abs(xs[idx[-1]] - x0)
    if lam <= 0.0:
        lam = 1.0
    sw = sxw = syw = sxxw = sxyw = 0.0
    for i in idx:
        u = abs(xs[i] - x0) / lam
        w = _tricube(u)
        if rho is not None:
            w *= rho[i]
        if w <= 0.0:
            continue
        dx = xs[i] - x0
        sw += w
        sxw += w * dx
        syw += w * ys[i]
        sxyw += w * dx * ys[i]
        sxxw += w * dx * dx
    if sw <= 0.0:
        # all weights vanished: unweighted local mean fallback
        return sum(ys[i] for i in idx) / len(idx)
    if degree == 0:
        return syw / sw
    den = sw * sxxw - sxw * sxw
    if abs(den) < 1e-300:
        return syw / sw
    beta = (sw * sxyw - sxw * syw) / den
    alpha = (syw - beta * sxw) / sw
    return alpha  # value at dx = 0


def _ma(v, k):
    # moving average of length k; output length len(v) - k + 1
    out = []
    s = sum(v[:k])
    out.append(s / k)
    for i in range(k, len(v)):
        s += v[i] - v[i - k]
        out.append(s / k)
    return out


def _next_odd(v):
    v = int(math.ceil(v))
    return v if v % 2 == 1 else v + 1


def stl_decompose(x, period, s_window=7, t_window=None, l_window=None,
                  s_degree=1, t_degree=1, l_degree=1,
                  inner=2, outer=0):
    """
    STL: seasonal-trend decomposition based on loess. Full inner loop
    (Cleveland, Cleveland, McRae & Terpenning 1990, Sec. 2.3, Steps
    1-6) and outer robustness loop (Sec. 2.4). Returns dict with
    seasonal, trend, remainder, weights.
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    ys = [float(v) for v in xv]
    N = len(ys)
    np_ = int(period)
    if np_ < 2 or N < 2 * np_:
        raise ValueError("need period >= 2 and at least two full cycles")
    n_s = int(s_window)
    if n_s % 2 == 0:
        n_s += 1
    n_t = int(t_window) if t_window is not None else _next_odd(
        1.5 * np_ / (1.0 - 1.5 / n_s))
    if n_t % 2 == 0:
        n_t += 1
    n_l = int(l_window) if l_window is not None else _next_odd(np_)
    if n_l % 2 == 0:
        n_l += 1
    T = [0.0] * N
    S = [0.0] * N
    rho = [1.0] * N
    for it_outer in range(outer + 1):
        use_rho = rho if it_outer > 0 else None
        for _ in range(inner):
            # Step 1: detrending
            det = [ys[v] - T[v] for v in range(N)]
            # Step 2: cycle-subseries smoothing, extended one
            # position before and after each subseries -> C of
            # length N + 2 * np_
            C = [0.0] * (N + 2 * np_)
            for p in range(np_):
                pos = list(range(p, N, np_))
                ncs = len(pos)
                sxs = [float(i + 1) for i in range(ncs)]
                sys_ = [det[v] for v in pos]
                srho = [rho[v] for v in pos] if use_rho is not None else None
                for j in range(0, ncs + 2):
                    val = _loess_at(sxs, sys_, float(j), n_s, s_degree,
                                    srho)
                    # j = 0 maps to position p - np_ in the extended
                    # series (index p in C since C starts at -np_+...)
                    C[p + j * np_] = val
            # Step 3: low-pass filter of C: MA(np_), MA(np_), MA(3),
            # then loess q = n_l, d = l_degree
            L1 = _ma(C, np_)
            L2 = _ma(L1, np_)
            L3 = _ma(L2, 3)
            lxs = [float(i + 1) for i in range(len(L3))]
            L = [_loess_at(lxs, L3, float(v + 1), n_l, l_degree)
                 for v in range(N)]
            # Step 4: detrending of smoothed cycle-subseries
            S = [C[np_ + v] - L[v] for v in range(N)]
            # Step 5: deseasonalising
            des = [ys[v] - S[v] for v in range(N)]
            # Step 6: trend smoothing
            txs = [float(v + 1) for v in range(N)]
            T = [_loess_at(txs, des, float(v + 1), n_t, t_degree,
                           use_rho) for v in range(N)]
        R = [ys[v] - T[v] - S[v] for v in range(N)]
        if it_outer < outer:
            # robustness weights: h = 6 median|R|, rho = B(|R|/h)
            ar = sorted(abs(r) for r in R)
            mid = N // 2
            med = ar[mid] if N % 2 == 1 else 0.5 * (ar[mid - 1] + ar[mid])
            h = 6.0 * med
            if h <= 0.0:
                rho = [1.0] * N
            else:
                rho = []
                for r in R:
                    u = abs(r) / h
                    if u >= 1.0:
                        rho.append(0.0)
                    else:
                        t = 1.0 - u * u
                        rho.append(t * t)
    return {"seasonal": S, "trend": T, "remainder": R, "weights": rho,
            "s_window": n_s, "t_window": n_t, "l_window": n_l}


def stlAn(x, period, s_window=7, k=3.0, inner=2, outer=0,
          t_window=None, l_window=None):
    """
    STL decomposition with residual-based outlier flags.

    Decomposes Y_v = T_v + S_v + R_v by STL (Cleveland, Cleveland,
    McRae & Terpenning 1990): inner loop Steps 1-6 of their Sec. 2.3
    (detrending; loess cycle-subseries smoothing with span n_s
    extended one cycle each side; low-pass filtering of the smoothed
    cycle-subseries by two MA(n_p), an MA(3) and a loess with span
    n_l; detrending of the smoothed cycle-subseries; deseasonalising;
    trend loess with span n_t) and, when `outer` > 0, the robustness
    iterations of Sec. 2.4 with bisquare weights rho_v = B(|R_v| / h),
    h = 6 median|R_v|. Defaults n_t = next odd >=
    1.5 n_p / (1 - 1.5/n_s) and n_l = next odd >= n_p follow their
    Secs. 3.4 and 3.6 recommendations; n_i = 2, n_o = 0 follow
    Sec. 3.3.

    Outliers are then flagged where the remainder deviates from its
    median by more than k robust standard deviations,
    sigma_hat = 1.4826 MAD (the MAD-to-sigma factor b = 1.4826 for
    Gaussian data; Hochenbaum, Vallis & Kejariwal 2017, eq (7)-(8)).

    Parameters
    ----------
    x : array-like
        Series (complete, at least two full cycles).
    period : int
        n_p, observations per seasonal cycle.
    s_window : int
        Seasonal loess span n_s (odd, >= 7 recommended, Sec. 3.5).
    k : float
        Outlier threshold in robust sigmas.
    inner, outer : int
        n_i inner passes and n_o robustness iterations.
    t_window, l_window : int, optional
        Trend and low-pass spans (defaults above).

    Returns
    -------
    result : RichResult
        Keys: seasonal, trend, remainder, outliers (1-based
        positions), threshold, sigma_hat.

    References
    ----------
    Cleveland, R. B., Cleveland, W. S., McRae, J. E. and Terpenning,
    I. (1990), "STL: a seasonal-trend decomposition procedure based
    on loess", Journal of Official Statistics 6(1), 3-73. Secs.
    2.1-2.4 (loess, inner loop Steps 1-6, outer loop), Secs. 3.3-3.6
    (parameter defaults).
    Hochenbaum, J., Vallis, O. S. and Kejariwal, A. (2017),
    "Automatic anomaly detection in the cloud via statistical
    learning", arXiv:1704.07706, eq (7)-(8) (MAD scaling).
    Source PDFs: /run/media/rootcoder/WD_BLACK/library/pdf/
    fetched-wave3/cleveland-cleveland-mcrae-terpenning-1990-stl-jos.pdf
    and hochenbaum-vallis-kejariwal-2017-twitter-shesd-anomaly-
    arxiv1704.07706.pdf
    """
    fit = stl_decompose(x, period, s_window=s_window, t_window=t_window,
                        l_window=l_window, inner=inner, outer=outer)
    R = fit["remainder"]
    N = len(R)
    sr = sorted(R)
    mid = N // 2
    med = sr[mid] if N % 2 == 1 else 0.5 * (sr[mid - 1] + sr[mid])
    ad = sorted(abs(r - med) for r in R)
    mad = ad[mid] if N % 2 == 1 else 0.5 * (ad[mid - 1] + ad[mid])
    sigma = 1.4826 * mad
    thr = k * sigma
    outl = [v + 1 for v in range(N) if abs(R[v] - med) > thr] \
        if sigma > 0.0 else []
    return RichResult(payload={
        "seasonal": fit["seasonal"],
        "trend": fit["trend"],
        "remainder": R,
        "outliers": outl,
        "threshold": thr,
        "sigma_hat": sigma,
        "s_window": fit["s_window"],
        "t_window": fit["t_window"],
        "l_window": fit["l_window"],
        "estimate": outl,
        "n": N,
        "method": "STL + MAD residual outliers (Cleveland et al. 1990)",
    })


def stl_anomaly(x, period, **kw):
    """Alias for stlAn (original stub export name)."""
    return stlAn(x, period, **kw)


def cheatsheet():
    return "stlAn(x, period) -> STL seasonal/trend/remainder + MAD outlier flags"
