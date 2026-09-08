# morie.fn -- function file (rootcoder007/morie)
"""MSE-optimal bandwidth selector for RDD (Imbens-Kalyanaraman)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mse_optimal_bandwidth_rdd"]

# Edge (triangular) kernel constant, IK (2012) sec. 4.2 and eq. (3.5).
CK_EDGE = 3.4375


def _solve_sym(A, b):
    """Gaussian elimination with partial pivoting on a small dense system.

    Written out rather than routed through the shared least-squares
    helper on purpose: the Python helper is a modified Gram-Schmidt QR
    and the R helper is an SVD, and two different factorisations of the
    same design do not agree to the last digits.  IK's algorithm is
    itself written in normal-equation form (lambda-hat = (T'T)^-1 T'Y),
    so this is the arithmetic the paper prescribes, and the identical
    loop in both arms makes the two agree exactly.
    """
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for k in range(n):
        piv = k
        best = abs(M[k][k])
        for i in range(k + 1, n):
            if abs(M[i][k]) > best:
                best = abs(M[i][k])
                piv = i
        if best < 1e-300:
            raise ValueError("mse_optimal_bandwidth_rdd: singular design in a pilot regression")
        if piv != k:
            M[k], M[piv] = M[piv], M[k]
        pk = M[k][k]
        for i in range(k + 1, n):
            f = M[i][k] / pk
            if f != 0.0:
                for j in range(k, n + 1):
                    M[i][j] -= f * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def _ols(rows, y):
    """Least squares through the normal equations, small p."""
    p = len(rows[0])
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for r, yi in zip(rows, y):
        for i in range(p):
            b[i] += r[i] * yi
            for j in range(p):
                A[i][j] += r[i] * r[j]
    return _solve_sym(A, b)


def _median(v):
    """Median with the even case averaged, as IK (2012) p.9 specifies."""
    s = sorted(v)
    m = len(s)
    if m == 0:
        raise ValueError("mse_optimal_bandwidth_rdd: median of an empty side")
    h = m // 2
    return s[h] if m % 2 == 1 else 0.5 * (s[h - 1] + s[h])


def ik_hopt(sigma2, f_hat, m2_plus, m2_minus, r_plus, r_minus, n, ck=CK_EDGE):
    """Step 3 of IK (2012): combine the plug-ins into the bandwidth.

    Formula, IK (2012) eq. (4.7) p.8::

        h_opt = C_K * ( (2 sigma^2(c) / f(c))
                        / ( (m2_plus - m2_minus)^2 + (r_plus + r_minus) )
                      )^(1/5) * N^(-1/5)

    Exposed separately so it can be checked against the paper's own
    worked example (p.15-16) without re-running the plug-in steps.
    """
    denom = (m2_plus - m2_minus) ** 2 + (r_plus + r_minus)
    if f_hat <= 0.0 or denom <= 0.0 or n <= 0:
        raise ValueError("mse_optimal_bandwidth_rdd: degenerate bandwidth criterion")
    return ck * ((2.0 * sigma2 / f_hat) / denom) ** 0.2 * float(n) ** -0.2


def mse_optimal_bandwidth_rdd(y, x, cutoff=0.0, kernel_constant=CK_EDGE):
    """Plug-in MSE-optimal bandwidth for a sharp RD design.

    A local-linear RD estimator trades squared bias, which grows with the
    window, against variance, which shrinks with it.  IK's selector is
    the minimiser of that sum with the six unknown population quantities
    replaced by plug-ins, plus a regularisation term that keeps the
    answer finite when the two curvatures happen to cancel -- without it
    the criterion has a pole at ``m2_plus == m2_minus`` and the selected
    window runs away to the whole support.

    The three steps are the paper's own (sec. 4.2, pp.9-10):

    1. Silverman-type pilot ``h1 = 1.84 * S_X * N^(-1/5)``, then the
       density ``f(c)`` and residual variance ``sigma^2(c)`` at the
       cutoff from the units inside it.
    2. A global cubic with a jump gives ``m3``, which sets the pilot
       bandwidths ``h2+`` and ``h2-`` (eq. 4.11); a local quadratic on
       each of those windows gives the curvatures ``m2+``, ``m2-``.
    3. Regularisation terms ``r+``, ``r-`` (eq. 4.12) and the
       combination in eq. (4.7).

    Note on eq. (4.8).  The density estimator is printed on p.9 as
    ``(N_h1- + N_h1+) / (N * h1)``, but the paper's own worked example on
    p.15 evaluates ``(836 + 862) / (2 * 6558 * 0.1445) = 0.8962``.  The
    factor two is required -- the window is the interval of width
    ``2 * h1`` around the cutoff -- and the printed eq. (4.8) is missing
    it.  This implementation follows the worked example, which
    reproduces the paper's own reported numbers; without the two the
    density comes out at 1.79 and the reported ``h_opt = 0.2649`` is not
    recoverable.

    Parameters
    ----------
    y : array-like
        Outcome.
    x : array-like
        Running variable, same length as ``y``.
    cutoff : float, default 0.0
        Threshold ``c``.
    kernel_constant : float, default 3.4375
        ``C_K``.  3.4375 is the edge (triangular) kernel value stated on
        p.10; 5.4 is the uniform-kernel value.

    Returns
    -------
    RichResult
        ``estimate`` and ``h_opt`` (the regularised bandwidth),
        ``h_no_reg``, ``h1``, ``f_hat``, ``sigma2``, ``m3``, ``h2_plus``,
        ``h2_minus``, ``m2_plus``, ``m2_minus``, ``r_plus``, ``r_minus``,
        ``n_plus``, ``n_minus``, ``n2_plus``, ``n2_minus``, ``n``.

    References
    ----------
    Imbens, G. & Kalyanaraman, K. (2012).  Optimal bandwidth choice for
    the regression discontinuity estimator.  Review of Economic Studies
    79(3):933-959.  doi:10.1093/restud/rdr043.  Equations and worked
    example read from the NBER working paper w14726 version, pp.8-10 and
    pp.15-16.
    """
    yv = [float(v) for v in C.vec(y)]
    xv = [float(v) for v in C.vec(x)]
    n = len(yv)
    if n == 0:
        raise ValueError("mse_optimal_bandwidth_rdd: y is empty")
    if len(xv) != n:
        raise ValueError("mse_optimal_bandwidth_rdd: x must have one entry per observation")
    c = float(cutoff)
    ck = float(kernel_constant)
    r = [xi - c for xi in xv]

    # ---- Step 1: pilot window, density and conditional variance at c
    sx = C.sd(xv, ddof=1)
    h1 = 1.84 * sx * float(n) ** -0.2
    if h1 <= 0.0:
        raise ValueError("mse_optimal_bandwidth_rdd: the running variable is constant")
    ip = [i for i in range(n) if 0.0 <= r[i] <= h1]
    im = [i for i in range(n) if -h1 <= r[i] < 0.0]
    n1p, n1m = len(ip), len(im)
    if n1p < 2 or n1m < 2:
        raise ValueError("mse_optimal_bandwidth_rdd: too few points inside the pilot window h1")
    ybp = sum(yv[i] for i in ip) / n1p
    ybm = sum(yv[i] for i in im) / n1m
    f_hat = (n1p + n1m) / (2.0 * n * h1)
    ss = sum((yv[i] - ybp) ** 2 for i in ip) + sum((yv[i] - ybm) ** 2 for i in im)
    sigma2 = ss / (n1p + n1m)
    if sigma2 <= 0.0:
        raise ValueError("mse_optimal_bandwidth_rdd: zero residual variance at the cutoff")

    # ---- Step 2: third derivative, pilot bandwidths, curvatures
    right = [i for i in range(n) if r[i] >= 0.0]
    left = [i for i in range(n) if r[i] < 0.0]
    n_plus, n_minus = len(right), len(left)
    if n_plus < 4 or n_minus < 4:
        raise ValueError("mse_optimal_bandwidth_rdd: each side needs at least four observations")
    med_p = _median([xv[i] for i in right])
    med_m = _median([xv[i] for i in left])
    keep = [i for i in range(n) if med_m <= xv[i] <= med_p]
    if len(keep) < 5:
        raise ValueError("mse_optimal_bandwidth_rdd: too few points for the global cubic")
    rows = [[1.0, 1.0 if r[i] >= 0.0 else 0.0, r[i], r[i] ** 2, r[i] ** 3] for i in keep]
    g = _ols(rows, [yv[i] for i in keep])
    m3 = 6.0 * g[4]

    base = (sigma2 / (f_hat * max(m3 * m3, 0.01))) ** (1.0 / 7.0)
    h2p = 3.56 * base * float(n_plus) ** (-1.0 / 7.0)
    h2m = 3.56 * base * float(n_minus) ** (-1.0 / 7.0)

    def _curv(idx, who):
        if len(idx) < 3:
            raise ValueError("mse_optimal_bandwidth_rdd: too few points for the " + who + " local quadratic")
        q = _ols([[1.0, r[i], r[i] ** 2] for i in idx], [yv[i] for i in idx])
        return 2.0 * q[2]

    i2p = [i for i in range(n) if 0.0 <= r[i] <= h2p]
    i2m = [i for i in range(n) if -h2m <= r[i] < 0.0]
    n2p, n2m = len(i2p), len(i2m)
    m2p = _curv(i2p, "right")
    m2m = _curv(i2m, "left")

    # ---- Step 3: regularisation and the bandwidth itself
    r_plus = 720.0 * sigma2 / (n2p * h2p ** 4)
    r_minus = 720.0 * sigma2 / (n2m * h2m ** 4)
    h_opt = ik_hopt(sigma2, f_hat, m2p, m2m, r_plus, r_minus, n, ck)
    h_no_reg = ik_hopt(sigma2, f_hat, m2p, m2m, 0.0, 0.0, n, ck)

    return RichResult(payload={
        "estimate": h_opt, "h_opt": h_opt, "h_no_reg": h_no_reg, "h1": h1,
        "f_hat": f_hat, "sigma2": sigma2, "m3": m3,
        "h2_plus": h2p, "h2_minus": h2m, "m2_plus": m2p, "m2_minus": m2m,
        "r_plus": r_plus, "r_minus": r_minus,
        "n_plus": n_plus, "n_minus": n_minus, "n2_plus": n2p, "n2_minus": n2m,
        "n1_plus": n1p, "n1_minus": n1m, "n": n, "ck": ck,
        "method": "IK (2012) MSE-optimal RDD bandwidth"})


def cheatsheet():
    return "rdmcbw: IK (2012) plug-in MSE-optimal bandwidth for sharp RDD"
