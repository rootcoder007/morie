# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kernel change-point analysis (regularised kernel Fisher discriminant scan)."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kcusum", "kernel_cusum", "kernelcusum"]


def _gram(z, kernel, bandwidth):
    n = len(z)
    K = np.zeros((n, n))
    if kernel == "linear":
        for i in range(n):
            for j in range(i, n):
                if isinstance(z[i], list):
                    v = 0.0
                    for a, b in zip(z[i], z[j]):
                        v += a * b
                else:
                    v = z[i] * z[j]
                K[i, j] = v
                K[j, i] = v
        return K, None
    if kernel == "gaussian":
        d2 = [[0.0] * n for _ in range(n)]
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                if isinstance(z[i], list):
                    s = 0.0
                    for a, b in zip(z[i], z[j]):
                        s += (a - b) * (a - b)
                else:
                    s = (z[i] - z[j]) * (z[i] - z[j])
                d2[i][j] = s
                d2[j][i] = s
                dists.append(math.sqrt(s))
        if bandwidth is None:
            ds = sorted(dists)
            m = len(ds)
            if m % 2 == 1:
                bandwidth = ds[m // 2]
            else:
                bandwidth = 0.5 * (ds[m // 2 - 1] + ds[m // 2])
            if bandwidth <= 0.0:
                bandwidth = 1.0
        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                v = math.exp(-d2[i][j] / (2.0 * bandwidth * bandwidth))
                K[i, j] = v
                K[j, i] = v
        return K, float(bandwidth)
    raise ValueError("kernel must be 'linear' or 'gaussian'")


def kcusum(x, kernel="gaussian", threshold=None, gamma=0.1,
           bandwidth=None, kmin=2, kmax=None):
    """
    Kernel change-point analysis (KCpA) running-maximum scan.

    Implements Harchaoui, Moulines & Bach (2008): for each candidate
    changepoint k the (maximum) kernel Fisher discriminant ratio,
    their Sec. 3,

      KFDR_{n,k;gamma} = (k (n-k) / n) *
        || (k/n Sigma_1:k + (n-k)/n Sigma_{k+1:n} + gamma I)^{-1/2}
           (mu_{k+1:n} - mu_{1:k}) ||^2_H,

    with empirical mean elements mu_{i:j} and covariance operators
    Sigma_{i:j} as defined in their Sec. 3 ("Kernel Fisher Discriminant
    Ratio" display). The scan statistic (their "Kernel change-point
    analysis" display) is

      T_{n;gamma}(k) = (KFDR_{n,k;gamma} - d_{1,n,k;gamma}) /
                       sqrt(2 d_{2,n,k;gamma}),
      d_1 = Tr[(Sigma_W + gamma I)^{-1} Sigma_W],
      d_2 = Tr[(Sigma_W + gamma I)^{-2} Sigma_W^2],
      n Sigma_W = k Sigma_{1:k} + (n-k) Sigma_{k+1:n},

    a studentisation making T(k) zero-mean unit-variance as n grows;
    the estimated changepoint is k_hat = argmax_k T(k) (running
    maximum partition strategy, their Sec. 2 and Figure 1).

    All operators are represented exactly on the span of the mapped
    sample via the Gram-matrix eigendecomposition; KFDR, d_1, d_2 are
    basis-invariant, so any orthonormal basis of the span gives the
    same values.

    Parameters
    ----------
    x : array-like
        Series (1-d) or matrix with observations in rows.
    kernel : str
        "gaussian" (median-heuristic bandwidth by default) or
        "linear" (in which case the statistic is a regularised version
        of the classical multivariate mean-change statistic; their
        Remark, Sec. 3).
    threshold : float, optional
        Decision threshold t_{1-alpha} for max_k T(k) (their
        Corollary 2); if given, `detected` is reported.
    gamma : float
        Ridge regularisation gamma > 0.
    bandwidth : float, optional
        Gaussian kernel bandwidth; default median heuristic.
    kmin, kmax : int, optional
        Scan interval [a_n, b_n], 1 < k < n (their Sec. 3, restriction
        away from the boundaries). Defaults 2 and n-2.

    Returns
    -------
    result : RichResult
        Keys: estimate (k_hat), statistic (max T), kfdr (at k_hat),
        d1, d2, T (full scan), detected (if threshold given),
        bandwidth, gamma.

    References
    ----------
    Harchaoui, Z., Moulines, E. and Bach, F. R. (2008), "Kernel
    change-point analysis", Advances in Neural Information Processing
    Systems 21 (NIPS 2008), pp. 609-616. Section 3 (KFDR and scan
    statistic definitions, d_1/d_2 normalisers), Section 2 (running
    maximum partition strategy), Corollary 2 (thresholding).
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    harchaoui-moulines-bach-2008-kernel-changepoint-analysis-nips.pdf
    """
    xv = np.asarray(x, dtype=float)
    if xv.ndim == 1:
        z = [float(v) for v in xv]
    else:
        z = [[float(v) for v in row] for row in xv]
    n = len(z)
    if n < 4:
        raise ValueError("need n >= 4")
    if kmax is None:
        kmax = n - 2
    kmin = int(kmin)
    kmax = int(kmax)
    if not (1 < kmin <= kmax < n):
        raise ValueError("need 1 < kmin <= kmax < n")
    K, bw = _gram(z, kernel, bandwidth)
    # orthonormal coordinates of the mapped sample: K = U L U';
    # coords R = diag(sqrt(l)) U' restricted to l > tol.
    lam, U = np.linalg.eigh(K)
    lmax = float(np.max(np.abs(lam)))
    tol = 1e-12 * (lmax if lmax > 0 else 1.0)
    keep = [i for i in range(n) if float(lam[i]) > tol]
    r = len(keep)
    C = np.zeros((r, n))
    for a, i in enumerate(keep):
        s = math.sqrt(float(lam[i]))
        for j in range(n):
            C[a, j] = s * float(U[j, i])
    Ts = []
    kf_all = []
    d1_all = []
    d2_all = []
    for k in range(kmin, kmax + 1):
        mu1 = np.mean(C[:, :k], axis=1)
        mu2 = np.mean(C[:, k:], axis=1)
        delta = mu2 - mu1
        A1 = C[:, :k] - mu1.reshape((r, 1))
        A2 = C[:, k:] - mu2.reshape((r, 1))
        S1 = (A1 @ A1.T) / float(k)
        S2 = (A2 @ A2.T) / float(n - k)
        Sw = (k * S1 + (n - k) * S2) / float(n)
        M = Sw + gamma * np.eye(r)
        sol = np.linalg.solve(M, delta)
        kfdr = (k * (n - k) / float(n)) * float(delta @ sol)
        Minv_S = np.linalg.solve(M, Sw)
        d1 = float(np.trace(Minv_S))
        Minv2_S2 = np.linalg.solve(M, np.linalg.solve(M, Sw @ Sw))
        d2 = float(np.trace(Minv2_S2))
        T = (kfdr - d1) / math.sqrt(2.0 * d2)
        Ts.append(T)
        kf_all.append(kfdr)
        d1_all.append(d1)
        d2_all.append(d2)
    ib = 0
    for i in range(1, len(Ts)):
        if Ts[i] > Ts[ib]:
            ib = i
    khat = kmin + ib
    out = {
        "estimate": int(khat),
        "statistic": float(Ts[ib]),
        "kfdr": float(kf_all[ib]),
        "d1": float(d1_all[ib]),
        "d2": float(d2_all[ib]),
        "T": [float(v) for v in Ts],
        "kmin": kmin,
        "kmax": kmax,
        "gamma": float(gamma),
        "bandwidth": bw,
        "n": n,
        "method": "Kernel change-point analysis (Harchaoui-Moulines-Bach 2008)",
    }
    if threshold is not None:
        out["threshold"] = float(threshold)
        out["detected"] = bool(Ts[ib] > threshold)
    return RichResult(payload=out)


def kernel_cusum(x, kernel="gaussian", threshold=None, **kw):
    """Alias for kcusum (original stub export name)."""
    return kcusum(x, kernel=kernel, threshold=threshold, **kw)


kernelcusum = kernel_cusum


def cheatsheet():
    return "kcusum(x, kernel, threshold) -> KFDR running-maximum changepoint scan"
