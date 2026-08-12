"""Unscented Kalman filter (Julier & Uhlmann 1997)."""

import math

from ._richresult import RichResult

__all__ = ["ukfF", "unscented_kalman_filter"]


def _chol(a):
    k = len(a)
    l = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1):
            s = sum(l[i][t] * l[j][t] for t in range(j))
            if i == j:
                v = a[i][i] - s
                if v < -1e-10:
                    raise ValueError("covariance not positive semidefinite")
                l[i][j] = math.sqrt(max(v, 0.0))
            else:
                l[i][j] = (a[i][j] - s) / l[j][j] if l[j][j] > 0 else 0.0
    return l


def _solve_mat(a, b_cols):
    # solve A X = B for X (small systems, partial pivoting)
    k = len(a)
    m = [row[:] + b[:] for row, b in zip(a, b_cols)]
    nb = len(b_cols[0])
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            raise ValueError("singular innovation covariance")
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        for j in range(k + nb):
            m[c][j] /= d
        for r in range(k):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                for j in range(k + nb):
                    m[r][j] -= f * m[c][j]
    return [row[k:] for row in m]


def _sigma_points(x, P, kappa):
    n = len(x)
    scale = n + kappa
    a = [[scale * P[i][j] for j in range(n)] for i in range(n)]
    l = _chol(a)
    pts = [list(x)]
    w = [kappa / scale]
    for i in range(n):
        col = [l[r][i] for r in range(n)]
        pts.append([x[r] + col[r] for r in range(n)])
        pts.append([x[r] - col[r] for r in range(n)])
        w.extend([1.0 / (2.0 * scale)] * 2)
    return pts, w


def _ut(pts, w, fun):
    ys = [list(fun(p)) for p in pts]
    m = len(ys[0])
    mean = [sum(wi * y[r] for wi, y in zip(w, ys)) for r in range(m)]
    cov = [[sum(wi * (y[a] - mean[a]) * (y[b] - mean[b])
                for wi, y in zip(w, ys)) for b in range(m)]
           for a in range(m)]
    return ys, mean, cov


def ukfF(f, h, Q, R, x0, P0, measurements, kappa=None):
    """
    Unscented Kalman filter for additive-noise nonlinear systems.

    Julier & Uhlmann (1997): the n-dimensional state (mean x,
    covariance P) is represented by 2n + 1 deterministic sigma
    points (their Eq. 12): X_0 = x with weight kappa/(n + kappa),
    X_i = x +/- column i of the matrix square root of (n + kappa) P
    with weights 1/(2(n + kappa)); any matrix square root works and
    the Cholesky factor is used (their remark 2).  Each step
    propagates the points through the process model f, forms the
    predicted mean/covariance as the weighted statistics of the
    transformed points (Eq. 13-14) plus Q, then transforms through
    the measurement model h and applies the standard Kalman update
    with gain K = P_xy P_yy^{-1}.  The unscented transform is exact
    for linear f and h, so the filter reproduces the linear Kalman
    filter exactly in that case.

    Sources
    -------
    Julier, S. J. & Uhlmann, J. K. (1997). A new extension of the
    Kalman filter to nonlinear systems. *Proc. SPIE 3068, Signal
    Processing, Sensor Fusion, and Target Recognition VI*, Eqs.
    12-14 and the transformation procedure (local copy
    fetched-wave3/julier-uhlmann-1997-ukf.pdf).

    Parameters
    ----------
    f : callable
        Process model x_{k+1} = f(x_k) (list -> list).
    h : callable
        Measurement model z_k = h(x_k) (list -> list).
    Q, R : matrices
        Additive process and measurement noise covariances.
    x0, P0 : vector, matrix
        Initial state mean and covariance.
    measurements : sequence of vectors
        Observations z_1, ..., z_T.
    kappa : float, optional
        Sigma-point spread parameter (default 3 - n, the paper's
        recommendation for Gaussian priors, but never below
        -n + 1e-6; 0 if that would be negative and n < 3).

    Returns
    -------
    RichResult
        Keys: states (filtered means), covariances, innovations,
        kappa.
    """
    x = [float(v) for v in x0]
    n = len(x)
    P = [[float(v) for v in row] for row in P0]
    Q = [[float(v) for v in row] for row in Q]
    R = [[float(v) for v in row] for row in R]
    if kappa is None:
        kappa = 3.0 - n
        if n + kappa <= 0:
            kappa = 1e-6 - n + 1.0
    kappa = float(kappa)
    if n + kappa <= 0:
        raise ValueError("need n + kappa > 0")
    states, covs, innovs = [], [], []
    for z in measurements:
        z = [float(v) for v in z]
        # predict
        pts, w = _sigma_points(x, P, kappa)
        _, xp, Pp = _ut(pts, w, f)
        for a in range(n):
            for b in range(n):
                Pp[a][b] += Q[a][b]
        # update: fresh sigma points from the predicted distribution
        pts2, w2 = _sigma_points(xp, Pp, kappa)
        ys, zp, Pzz = _ut(pts2, w2, h)
        m = len(zp)
        for a in range(m):
            for b in range(m):
                Pzz[a][b] += R[a][b]
        Pxz = [[sum(wi * (p[a] - xp[a]) * (y[b] - zp[b])
                    for wi, p, y in zip(w2, pts2, ys))
                for b in range(m)] for a in range(n)]
        # K = Pxz Pzz^{-1}  <=>  solve Pzz K' = Pxz'
        kt = _solve_mat(Pzz, [[Pxz[a][b] for a in range(n)]
                              for b in range(m)])
        K = [[kt[b][a] for b in range(m)] for a in range(n)]
        innov = [z[b] - zp[b] for b in range(m)]
        x = [xp[a] + sum(K[a][b] * innov[b] for b in range(m))
             for a in range(n)]
        P = [[Pp[a][b] - sum(K[a][c] * Pzz[c][d] * K[b][d]
                             for c in range(m) for d in range(m))
              for b in range(n)] for a in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                v = 0.5 * (P[a][b] + P[b][a])
                P[a][b] = P[b][a] = v
        states.append(list(x))
        covs.append([row[:] for row in P])
        innovs.append(innov)
    return RichResult(payload={
        "states": states,
        "covariances": covs,
        "innovations": innovs,
        "kappa": kappa,
        "method": "unscented Kalman filter (Julier & Uhlmann 1997)",
    })


# long descriptive alias (stub-era name)
unscented_kalman_filter = ukfF


def cheatsheet():
    return "ukfF: 2n+1 sigma points, UT predict + Kalman gain update"
