# morie.fn -- function file (rootcoder007/morie)
"""Empirical Mode Decomposition."""

from ._richresult import RichResult

__all__ = ["emd_decomposition"]


def _natural_spline(xk, yk, xq):
    """Natural cubic spline through (xk, yk), evaluated at xq."""
    m = len(xk)
    if m == 1:
        return [yk[0]] * len(xq)
    if m == 2:
        s = (yk[1] - yk[0]) / (xk[1] - xk[0])
        return [yk[0] + s * (t - xk[0]) for t in xq]
    h = [xk[i + 1] - xk[i] for i in range(m - 1)]
    # Tridiagonal system for the second derivatives, natural ends c0 = cm = 0.
    a = [0.0] * m
    b = [0.0] * m
    c = [0.0] * m
    d = [0.0] * m
    b[0] = 1.0
    b[m - 1] = 1.0
    for i in range(1, m - 1):
        a[i] = h[i - 1]
        b[i] = 2.0 * (h[i - 1] + h[i])
        c[i] = h[i]
        d[i] = 3.0 * ((yk[i + 1] - yk[i]) / h[i] - (yk[i] - yk[i - 1]) / h[i - 1])
    # Thomas algorithm
    cp = [0.0] * m
    dp = [0.0] * m
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, m):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i] * dp[i - 1]) / den
    C = [0.0] * m
    C[m - 1] = dp[m - 1]
    for i in range(m - 2, -1, -1):
        C[i] = dp[i] - cp[i] * C[i + 1]
    B = [0.0] * (m - 1)
    D = [0.0] * (m - 1)
    for i in range(m - 1):
        B[i] = (yk[i + 1] - yk[i]) / h[i] - h[i] * (2.0 * C[i] + C[i + 1]) / 3.0
        D[i] = (C[i + 1] - C[i]) / (3.0 * h[i])
    out = []
    for t in xq:
        if t <= xk[0]:
            i = 0
        elif t >= xk[m - 1]:
            i = m - 2
        else:
            i = 0
            while i < m - 2 and xk[i + 1] <= t:
                i += 1
        u = t - xk[i]
        out.append(yk[i] + B[i] * u + C[i] * u * u + D[i] * u * u * u)
    return out


def _extrema(v):
    n = len(v)
    hi = []
    lo = []
    for i in range(1, n - 1):
        if v[i] > v[i - 1] and v[i] >= v[i + 1]:
            hi.append(i)
        if v[i] < v[i - 1] and v[i] <= v[i + 1]:
            lo.append(i)
    return hi, lo


def emd_decomposition(y, max_imf=10, max_sift=50, sd_tol=0.2):
    """
    Empirical Mode Decomposition

    Formula: iterative sifting into intrinsic mode functions.  Given the
    running residual r, the sifting loop of Huang et al (1998) sec. 4 is

        1. locate the local maxima and minima of h
        2. cubic-spline the maxima into an upper envelope u and the
           minima into a lower envelope l
        3. m = (u + l) / 2 ; h <- h - m
        4. stop when SD = sum (h_k - h_{k-1})^2 / sum h_{k-1}^2 < sd_tol
           (Huang eq. 5.5, recommended 0.2-0.3)

    The extracted h is an IMF; it is subtracted from r and the loop
    repeats until the residual has fewer than three extrema (it is
    monotone or a single hump) or ``max_imf`` IMFs have been taken.
    Endpoints are appended to both extremum sets so the envelopes span
    the record, which is the usual simple end condition.

    Parameters
    ----------
    y : array-like
        Signal.
    max_imf : int
        Maximum number of IMFs to extract.
    max_sift : int
        Maximum sifting iterations per IMF.
    sd_tol : float
        Cauchy-type stopping threshold SD.

    Returns
    -------
    result : dict
        Keys: estimate (number of IMFs), n_imf, imfs (row-major, one
        row per IMF), residual, completeness (max |sum of parts - y|),
        n, method.

    References
    ----------
    Huang, Shen, Long, Wu, Shih, Zheng, Yen, Tung & Liu (1998),
    Proc. R. Soc. Lond. A 454(1971):903-995, doi:10.1098/rspa.1998.0193.
    """
    y = [float(v) for v in y]
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    max_imf = int(max_imf)
    if max_imf < 1:
        raise ValueError("max_imf must be positive")
    if float(sd_tol) <= 0.0:
        raise ValueError("sd_tol must be positive")
    x = [float(i) for i in range(n)]
    r = list(y)
    imfs = []
    while len(imfs) < max_imf:
        hi, lo = _extrema(r)
        if len(hi) + len(lo) < 3:
            break
        h = list(r)
        for _ in range(int(max_sift)):
            hi, lo = _extrema(h)
            if len(hi) == 0 or len(lo) == 0:
                break
            xu = [0] + hi + [n - 1]
            xl = [0] + lo + [n - 1]
            up = _natural_spline([x[i] for i in xu], [h[i] for i in xu], x)
            dn = _natural_spline([x[i] for i in xl], [h[i] for i in xl], x)
            prev = h
            h = [prev[i] - 0.5 * (up[i] + dn[i]) for i in range(n)]
            den = sum(v * v for v in prev)
            if den <= 0.0:
                break
            sd = sum((h[i] - prev[i]) ** 2 for i in range(n)) / den
            if sd < float(sd_tol):
                break
        imfs.append(h)
        r = [r[i] - h[i] for i in range(n)]
    tot = [sum(f[i] for f in imfs) + r[i] for i in range(n)]
    comp = max(abs(tot[i] - y[i]) for i in range(n))
    return RichResult(payload={
        "estimate": float(len(imfs)),
        "n_imf": len(imfs),
        "imfs": [v for f in imfs for v in f],
        "residual": r,
        "completeness": comp,
        "n": n,
        "method": "Empirical Mode Decomposition",
    })


def cheatsheet():
    return "emdtsm: Empirical Mode Decomposition"


# compact alias per ledger/NAMING.md
emddecomposition = emd_decomposition
