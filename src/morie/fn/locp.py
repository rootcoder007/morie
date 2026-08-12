"""Local polynomial regression smoother (Fan & Gijbels 1996; ESL Sec. 6.1)."""

import math

from ._richresult import RichResult

__all__ = ["locp", "local_polynomial_smoother"]


def _kernel(name, t):
    # t = |x - x0| / lambda
    if t >= 1.0 and name != "gaussian":
        return 0.0
    if name == "tricube":
        return (1.0 - t ** 3) ** 3
    if name == "epanechnikov":
        return 0.75 * (1.0 - t * t)
    if name == "gaussian":
        return math.exp(-0.5 * t * t)
    raise ValueError("kernel must be tricube, epanechnikov or gaussian")


def _solve(a, b):
    # Gaussian elimination with partial pivoting on the (small) normal
    # equations; returns None on singularity.
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c] / m[c][c]
                for j in range(c, n + 1):
                    m[r][j] -= f * m[c][j]
    return [m[i][n] / m[i][i] for i in range(n)]


def locp(x, y, x0=None, degree=1, bandwidth=None, kernel="tricube"):
    """
    Local polynomial regression smoother.

    At each evaluation point x0 solve the kernel-weighted least
    squares problem of ESL Eq. 6.11 (Hastie, Tibshirani & Friedman),
    the estimator studied systematically by Fan & Gijbels (1996):

        min_{alpha, beta_j} sum_i K_lambda(x0, x_i)
            [ y_i - alpha - sum_{j=1}^d beta_j (x_i - x0)^j ]^2,

    and report f_hat(x0) = alpha_hat (centered parameterization, so
    the intercept is the fit and beta_1 the local slope).  Degree 1
    is local linear regression, which corrects the boundary bias of
    kernel averages (ESL Sec. 6.1.1); a polynomial of degree d is
    reproduced exactly for any kernel and bandwidth.

    Sources
    -------
    Fan, J. & Gijbels, I. (1996). *Local Polynomial Modelling and
    Its Applications*. Chapman & Hall (the estimator's standard
    monograph, as cited by the stub).
    Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements
    of Statistical Learning*, 2nd ed., Springer, Sec. 6.1, Eqs.
    6.7-6.11 (local WLS formulation and kernels; local PDF:
    WD_BLACK/library/pdf/BookAdvanced_elementsofstatisticallearning.pdf).

    Parameters
    ----------
    x, y : sequences of float
        Observations.
    x0 : sequence of float, optional
        Evaluation points (default: the sorted unique x).
    degree : int
        Polynomial degree d >= 0 (0 = Nadaraya-Watson average).
    bandwidth : float, optional
        Kernel half-width lambda (metric); default = half the x
        range.
    kernel : str
        "tricube" (ESL default), "epanechnikov" or "gaussian".

    Returns
    -------
    RichResult
        Keys: fitted (f_hat at each x0), x0, slope (beta_1 at each
        x0 when degree >= 1), n_effective (sum of weights per x0).
    """
    xv = [float(v) for v in x]
    yv = [float(v) for v in y]
    n = len(xv)
    if len(yv) != n or n < 2:
        raise ValueError("x and y must be paired with n >= 2")
    d = int(degree)
    if d < 0:
        raise ValueError("degree must be >= 0")
    if x0 is None:
        pts = sorted(set(xv))
    else:
        pts = [float(v) for v in x0]
    if bandwidth is None:
        bandwidth = (max(xv) - min(xv)) / 2.0
    lam = float(bandwidth)
    if lam <= 0:
        raise ValueError("bandwidth must be positive")
    kern = str(kernel).lower()
    fitted = []
    slope = []
    neff = []
    for p0 in pts:
        w = [_kernel(kern, abs(xi - p0) / lam) for xi in xv]
        sw = sum(w)
        if sw <= 0 or sum(1 for v in w if v > 0) < d + 1:
            fitted.append(float("nan"))
            slope.append(float("nan"))
            neff.append(sw)
            continue
        # normal equations for centered design [1, (x-x0), ..., (x-x0)^d]
        a = [[0.0] * (d + 1) for _ in range(d + 1)]
        b = [0.0] * (d + 1)
        for xi, yi, wi in zip(xv, yv, w):
            if wi == 0.0:
                continue
            z = xi - p0
            pows = [1.0]
            for _ in range(2 * d):
                pows.append(pows[-1] * z)
            for r in range(d + 1):
                b[r] += wi * yi * pows[r]
                for c in range(d + 1):
                    a[r][c] += wi * pows[r + c]
        beta = _solve(a, b)
        if beta is None:
            fitted.append(float("nan"))
            slope.append(float("nan"))
        else:
            fitted.append(beta[0])
            slope.append(beta[1] if d >= 1 else float("nan"))
        neff.append(sw)
    return RichResult(payload={
        "fitted": fitted, "x0": pts, "slope": slope,
        "n_effective": neff, "degree": d, "bandwidth": lam,
        "kernel": kern,
        "method": "local polynomial WLS (Fan-Gijbels 1996; ESL Eq. 6.11)",
    })


# long descriptive alias (stub-era name)
local_polynomial_smoother = locp


def cheatsheet():
    return "locp: per-x0 kernel-WLS on centered polynomial; f_hat = alpha_hat"
