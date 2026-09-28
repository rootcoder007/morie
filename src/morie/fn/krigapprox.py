# morie.fn -- function file (rootcoder007/morie)
"""Scalable approximations to kriging: Vecchia likelihood, nearest-neighbour (NNGP) prediction, covariance
tapering, sparse Gaussian-process predictors (subset of regressors, DTC / predictive process, FITC) and
fixed rank kriging."""

from __future__ import annotations

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from .krgsys import kriging_covariance

__all__ = ["vecchia_loglik", "nngp_predict", "tapered_kriging", "sparse_gp_krige", "fixed_rank_kriging"]


def _rows(a):
    return [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in a]


def _dist(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += (x - y) * (x - y)
    return math.sqrt(s)


def _mv(M, v):
    return [ssum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _sk(zc, C, c0, c00):
    """Simple kriging on centred data: (prediction, variance)."""
    w = _mv(inverse(C), c0)
    return ssum(w[i] * zc[i] for i in range(len(zc))), c00 - ssum(w[i] * c0[i] for i in range(len(c0)))


def _nearest(P, q, idx, m):
    return sorted(idx, key=lambda j: (_dist(P[j], q), j))[:m]


def vecchia_loglik(z, coords, model, m: int = 10, *, mean: float = 0.0) -> RichResult:
    r"""Vecchia's (1988) approximate Gaussian log-likelihood from conditioning sets of ``m`` nearest predecessors.

    ``log L ~ sum_i log N(z_i; mu + c_i' C_{N_i}^{-1} (z_{N_i} - mu), C_ii - c_i' C_{N_i}^{-1} c_i)``
    where ``N_i`` holds the ``m`` observations nearest to ``s_i`` among
    ``s_1, ..., s_{i-1}`` (the data order is the ordering; sort the data,
    e.g. by a coordinate or maxmin, beforehand). ``m = n - 1`` recovers the
    exact likelihood. Covariance ``model`` as :func:`morie.fn.krgsys.kriging_covariance`.

    References
    ----------
    Vecchia, A. V. (1988). Estimation and model identification for continuous
    spatial processes. *Journal of the Royal Statistical Society B*, 50, 297-312.
    Katzfuss, M. and Guinness, J. (2021). A general framework for Vecchia
    approximations of Gaussian processes. *Statistical Science*, 36, 124-141.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 1.0, "nugget": 0.1}
    >>> round(vecchia_loglik([0.2, -0.1, 0.4], [(0, 0), (1, 0), (0, 1)], m, 2).loglik, 10)
    -2.8709746673
    """
    zv = [float(v) for v in z]
    P = _rows(coords)
    n = len(zv)
    ll = 0.0
    means, variances = [], []
    for i in range(n):
        nb = _nearest(P, P[i], list(range(i)), m)
        cii = kriging_covariance(0.0, model)
        if nb:
            C = [[kriging_covariance(_dist(P[a], P[b]), model) for b in nb] for a in nb]
            c = [kriging_covariance(_dist(P[a], P[i]), model) for a in nb]
            mu_c, v = _sk([zv[a] - mean for a in nb], C, c, cii)
            mu = mean + mu_c
        else:
            mu, v = mean, cii
        means.append(mu)
        variances.append(v)
        ll += -0.5 * math.log(2 * math.pi * v) - 0.5 * (zv[i] - mu) ** 2 / v
    return RichResult(payload={"loglik": ll, "conditional_mean": means, "conditional_variance": variances})


def nngp_predict(z, coords, new_coords, model, m: int = 10, *, mean: float | None = None) -> RichResult:
    r"""Nearest-neighbour Gaussian process (NNGP) kriging: each target conditions on its ``m`` nearest observations.

    Prediction ``mu + c_0' C_N^{-1} (z_N - mu)`` and variance
    ``C_00 - c_0' C_N^{-1} c_0`` from the ``m``-nearest neighbour set ``N``
    (Datta et al. 2016); ``mean`` defaults to the sample mean (plug-in).
    With ``m >= n`` this is simple kriging.

    References
    ----------
    Datta, A., Banerjee, S., Finley, A. O. and Gelfand, A. E. (2016).
    Hierarchical nearest-neighbor Gaussian process models for large
    geostatistical datasets. *JASA*, 111, 800-812.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    >>> r = nngp_predict([1.0, 2.0, 0.5, 1.5], [(0, 0), (1, 0), (0, 1), (5, 5)], [(0.2, 0.2)], m, 3, mean=1.0)
    >>> round(r.prediction[0], 10), r.neighbours[0]
    (1.0828596085, [1, 2, 3])
    """
    zv = [float(v) for v in z]
    P, Q = _rows(coords), _rows(new_coords)
    mu = ssum(zv) / len(zv) if mean is None else float(mean)
    pred, var, nbs = [], [], []
    for q in Q:
        nb = _nearest(P, q, list(range(len(P))), m)
        C = [[kriging_covariance(_dist(P[a], P[b]), model) for b in nb] for a in nb]
        c0 = [kriging_covariance(_dist(P[a], q), model) for a in nb]
        pk, vk = _sk([zv[a] - mu for a in nb], C, c0, kriging_covariance(0.0, model))
        pred.append(mu + pk)
        var.append(vk)
        nbs.append([a + 1 for a in nb])
    return RichResult(payload={"prediction": pred, "variance": var, "neighbours": nbs, "mean": mu})


def _wendland(h, theta):
    r = h / theta
    return (1 - r) ** 4 * (1 + 4 * r) if r < 1 else 0.0


def tapered_kriging(z, coords, new_coords, model, taper_range: float, *, mean: float | None = None) -> RichResult:
    r"""Simple kriging with a tapered covariance ``C(h) T(h)`` (Furrer, Genton and Nychka 2006).

    ``T`` is the Wendland taper ``(1 - h/theta)_+^4 (1 + 4h/theta)`` (positive
    definite in up to three dimensions), so the tapered covariance matrix is
    sparse (entries beyond ``taper_range`` vanish) and the tapered predictor
    is asymptotically optimal for Matern-type covariances. Both the data
    matrix and the target covariances are tapered. Returns predictions,
    the tapered kriging variances and the fraction of non-zero entries.

    References
    ----------
    Furrer, R., Genton, M. G. and Nychka, D. (2006). Covariance tapering for
    interpolation of large spatial datasets. *Journal of Computational and
    Graphical Statistics*, 15, 502-523.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 1.0}
    >>> r = tapered_kriging([1.0, 2.0, 0.5], [(0, 0), (1, 0), (0, 1)], [(0.2, 0.2)], m, 1.2, mean=1.0)
    >>> round(r.prediction[0], 10), round(r.density, 6)
    (1.0075610417, 0.777778)
    """
    zv = [float(v) for v in z]
    P, Q = _rows(coords), _rows(new_coords)
    n = len(zv)
    mu = ssum(zv) / n if mean is None else float(mean)

    def ct(h):
        return kriging_covariance(h, model) * _wendland(h, taper_range)

    C = [[ct(_dist(P[i], P[j])) for j in range(n)] for i in range(n)]
    nz = sum(1 for i in range(n) for j in range(n) if C[i][j] != 0.0)
    Ci = inverse(C)
    pred, var = [], []
    for q in Q:
        c0 = [ct(_dist(P[i], q)) for i in range(n)]
        w = _mv(Ci, c0)
        pred.append(mu + ssum(w[i] * (zv[i] - mu) for i in range(n)))
        var.append(ct(0.0) - ssum(w[i] * c0[i] for i in range(n)))
    return RichResult(payload={"prediction": pred, "variance": var, "density": nz / (n * n), "mean": mu})


def sparse_gp_krige(
    z, coords, new_coords, model, inducing, *, noise: float, method: str = "fitc", mean: float | None = None
) -> RichResult:
    r"""Low-rank kriging through ``k`` inducing points (knots): SoR, DTC (predictive process) and FITC.

    With ``Q_ab = K_au K_uu^{-1} K_ub`` the Nystrom (predictive-process)
    covariance, ``Lambda = noise I`` (``"sor"``, ``"dtc"``) or
    ``diag(K_ff - Q_ff) + noise I`` (``"fitc"``, the modified predictive
    process) and ``S = (K_uu + K_uf Lambda^{-1} K_fu)^{-1}``:
    mean ``mu + K_*u S K_uf Lambda^{-1} (y - mu)``; variance ``K_*u S K_u*``
    (SoR), ``K_** - Q_** + K_*u S K_u*`` (DTC, FITC). Predicts the noise-free
    signal. With the inducing set equal to the data, DTC and FITC are exact.

    References
    ----------
    Quinonero-Candela, J. and Rasmussen, C. E. (2005). A unifying view of
    sparse approximate Gaussian process regression. *JMLR*, 6, 1939-1959.
    Banerjee, S., Gelfand, A. E., Finley, A. O. and Sang, H. (2008).
    Gaussian predictive process models for large spatial data sets. *JRSS B*, 70, 825-848.

    Examples
    --------
    >>> m = {"model": "Exp", "psill": 1.0, "range": 2.0}
    >>> X = [(0, 0), (1, 0), (0, 1), (1, 1)]
    >>> r = sparse_gp_krige([1.0, 2.0, 0.5, 1.5], X, [(0.5, 0.5)], m, [(0, 0), (1, 1)], noise=0.1, mean=1.0)
    >>> round(r.prediction[0], 10)
    1.2267723058
    """
    if method not in ("sor", "dtc", "fitc"):
        raise ValueError("method must be 'sor', 'dtc' or 'fitc'")
    zv = [float(v) for v in z]
    P, Q, U = _rows(coords), _rows(new_coords), _rows(inducing)
    n, k = len(zv), len(U)
    mu = ssum(zv) / n if mean is None else float(mean)

    def kc(a, b):
        return kriging_covariance(_dist(a, b), model)

    Kuu = [[kc(U[a], U[b]) for b in range(k)] for a in range(k)]
    Kuui = inverse(Kuu)
    Kuf = [[kc(U[a], P[i]) for i in range(n)] for a in range(k)]
    lam = []
    for i in range(n):
        if method == "fitc":
            q = ssum(Kuf[a][i] * Kuui[a][b] * Kuf[b][i] for a in range(k) for b in range(k))
            lam.append(kc(P[i], P[i]) - q + noise)
        else:
            lam.append(noise)
    M = [[Kuu[a][b] + ssum(Kuf[a][i] * Kuf[b][i] / lam[i] for i in range(n)) for b in range(k)] for a in range(k)]
    S = inverse(M)
    v = [ssum(Kuf[a][i] * (zv[i] - mu) / lam[i] for i in range(n)) for a in range(k)]
    Sv = _mv(S, v)
    pred, var = [], []
    for q in Q:
        ku = [kc(U[a], q) for a in range(k)]
        pred.append(mu + ssum(ku[a] * Sv[a] for a in range(k)))
        Sk = _mv(S, ku)
        part = ssum(ku[a] * Sk[a] for a in range(k))
        if method == "sor":
            var.append(part)
        else:
            qss = ssum(ku[a] * Kuui[a][b] * ku[b] for a in range(k) for b in range(k))
            var.append(kc(q, q) - qss + part)
    return RichResult(payload={"prediction": pred, "variance": var, "mean": mu, "method": method})


def _bisquare(d, r):
    return (1 - (d / r) ** 2) ** 2 if d < r else 0.0


def fixed_rank_kriging(
    z, coords, new_coords, centers, radius, K, *, sigma2_eps: float, sigma2_xi: float = 0.0, mean: float | None = None
) -> RichResult:
    r"""Fixed rank kriging (Cressie and Johannesson 2008) with bisquare basis functions.

    ``Z(s) = mu + S(s)' eta + xi(s) + eps(s)`` with ``r`` basis functions
    ``S_l(s) = (1 - (||s - c_l|| / radius_l)^2)^2`` inside the radius,
    ``var(eta) = K`` (``r x r``), fine-scale variance ``sigma2_xi`` and
    measurement error ``sigma2_eps``. ``Sigma = S K S' + (sigma2_xi + sigma2_eps) I``
    is inverted by the Sherman-Morrison-Woodbury identity in ``O(n r^2)``:
    ``Sigma^{-1} = D^{-1} - D^{-1} S (K^{-1} + S' D^{-1} S)^{-1} S' D^{-1}``.
    Prediction of the smooth process ``mu + S_0' K S' Sigma^{-1} (Z - mu)``
    with mean squared prediction error ``S_0' K S_0 + sigma2_xi - S_0' K S' Sigma^{-1} S K S_0``.

    References
    ----------
    Cressie, N. and Johannesson, G. (2008). Fixed rank kriging for very large
    spatial data sets. *Journal of the Royal Statistical Society B*, 70, 209-226.

    Examples
    --------
    >>> r = fixed_rank_kriging([1.0, 2.0, 0.5, 1.5], [(0, 0), (1, 0), (0, 1), (1, 1)], [(0.5, 0.5)],
    ...                        [(0, 0), (1, 1)], 2.0, [[1.0, 0.3], [0.3, 1.0]], sigma2_eps=0.1, mean=1.0)
    >>> round(r.prediction[0], 10)
    1.312965453
    """
    zv = [float(v) for v in z]
    P, Q, Cn = _rows(coords), _rows(new_coords), _rows(centers)
    n, r = len(zv), len(Cn)
    rad = [float(radius)] * r if not isinstance(radius, (list, tuple)) else [float(v) for v in radius]
    mu = ssum(zv) / n if mean is None else float(mean)
    Kf = [[float(v) for v in row] for row in K]
    S = [[_bisquare(_dist(P[i], Cn[c_]), rad[c_]) for c_ in range(r)] for i in range(n)]
    d = sigma2_xi + sigma2_eps
    Ki = inverse(Kf)
    M = [[Ki[a][b] + ssum(S[i][a] * S[i][b] for i in range(n)) / d for b in range(r)] for a in range(r)]
    Mi = inverse(M)
    res = [v - mu for v in zv]
    StR = [ssum(S[i][a] * res[i] for i in range(n)) for a in range(r)]
    MiStR = _mv(Mi, StR)
    # Sigma^{-1} res = res / d - S Mi S' res / d^2
    sir = [res[i] / d - ssum(S[i][a] * MiStR[a] for a in range(r)) / (d * d) for i in range(n)]
    Stsir = [ssum(S[i][a] * sir[i] for i in range(n)) for a in range(r)]
    StS = [[ssum(S[i][a] * S[i][b] for i in range(n)) for b in range(r)] for a in range(r)]
    # S' Sigma^{-1} S = StS / d - StS Mi StS / d^2
    StSMi = [[ssum(StS[a][c] * Mi[c][b] for c in range(r)) for b in range(r)] for a in range(r)]
    G = [
        [StS[a][b] / d - ssum(StSMi[a][c] * StS[c][b] for c in range(r)) / (d * d) for b in range(r)] for a in range(r)
    ]
    KG = [[ssum(Kf[a][c] * G[c][b] for c in range(r)) for b in range(r)] for a in range(r)]
    KGK = [[ssum(KG[a][c] * Kf[c][b] for c in range(r)) for b in range(r)] for a in range(r)]
    KStsir = _mv(Kf, Stsir)
    pred, var = [], []
    for q in Q:
        s0 = [_bisquare(_dist(q, Cn[c_]), rad[c_]) for c_ in range(r)]
        pred.append(mu + ssum(s0[a] * KStsir[a] for a in range(r)))
        sks = ssum(s0[a] * Kf[a][b] * s0[b] for a in range(r) for b in range(r))
        red = ssum(s0[a] * KGK[a][b] * s0[b] for a in range(r) for b in range(r))
        var.append(sks + sigma2_xi - red)
    return RichResult(payload={"prediction": pred, "mspe": var, "basis": S, "mean": mu})


def cheatsheet() -> str:
    return "vecchia_loglik / nngp_predict / tapered_kriging / sparse_gp_krige / fixed_rank_kriging -> scalable kriging."
