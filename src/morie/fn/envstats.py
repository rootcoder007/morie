# morie.fn -- function file (rootcoder007/morie)
"""Climate and agreement statistics: accumulated cyclone energy, the linearised outgoing longwave radiation
of energy-balance models, prewhitened Mann-Kendall trend tests, Fleiss' kappa for many raters and the
empirical finite-sample breakdown point of an estimator."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult

__all__ = ["budyko_olr", "prewhitened_mann_kendall", "fleiss_kappa", "empirical_breakdown_point"]


def budyko_olr(T, *, A: float = 203.3, B: float = 2.09, S0: float = 1361.0, albedo: float = 0.3) -> RichResult:
    r"""Linearised outgoing longwave radiation ``OLR = A + B T`` (T in degrees C) and the energy-balance equilibrium.

    Budyko's (1969) parameterisation with North, Cahalan and Coakley's
    (1981) satellite-fitted constants ``A = 203.3 W m^-2``, ``B = 2.09
    W m^-2 C^-1`` by default; also returns the zero-dimensional equilibrium
    temperature ``T* = ((1 - albedo) S0 / 4 - A) / B`` and the climate
    sensitivity ``1 / B`` (C per W m^-2).

    References
    ----------
    Budyko, M. I. (1969). The effect of solar radiation variations on the
    climate of the Earth. *Tellus*, 21, 611-619.
    North, G. R., Cahalan, R. F. and Coakley, J. A. (1981). Energy balance
    climate models. *Reviews of Geophysics and Space Physics*, 19, 91-121.

    Examples
    --------
    >>> r = budyko_olr(15.0)
    >>> round(r.olr, 6), round(r.equilibrium_temperature, 6)
    (234.65, 16.686603)
    """
    t = float(T)
    return RichResult(
        payload={
            "olr": A + B * t,
            "equilibrium_temperature": ((1 - albedo) * S0 / 4 - A) / B,
            "sensitivity": 1.0 / B,
        }
    )


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _sen(x):
    return _median([(x[j] - x[i]) / (j - i) for i in range(len(x) - 1) for j in range(i + 1, len(x))])


def _acf1(x):
    m = ssum(x) / len(x)
    d = [v - m for v in x]
    return ssum(d[t] * d[t + 1] for t in range(len(x) - 1)) / ssum(v * v for v in d)


def _mk(y):
    n = len(y)
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += (y[j] > y[i]) - (y[j] < y[i])
    var = n * (n - 1) * (2 * n + 5) / 18.0
    counts = {}
    for v in y:
        counts[v] = counts.get(v, 0) + 1
    for t in counts.values():
        if t > 1:
            var -= t * (t - 1) * (2 * t + 5) / 18.0
    z = 0.0 if S == 0 else (S - 1) / math.sqrt(var) if S > 0 else (S + 1) / math.sqrt(var)
    return S, var, z


def prewhitened_mann_kendall(x, method: str = "tfpw") -> RichResult:
    r"""Mann-Kendall trend test after removing lag-1 autocorrelation (as ``modifiedmk::pwmk`` / ``tfpwmk``).

    ``"pw"`` (von Storch 1995): test ``x_t - r_1 x_{t-1}`` with ``r_1`` the
    lag-1 autocorrelation of ``x``. ``"tfpw"`` (trend-free prewhitening,
    Yue, Pilon, Phinney and Cavadias 2002): remove the Sen slope ``b``,
    prewhiten the detrended series, add ``b t`` back and test. ``S``,
    ``var(S)`` (tie-corrected), ``Z`` with continuity correction, two-sided
    ``p``, Kendall's tau and the Sen slope of the tested series.

    References
    ----------
    von Storch, H. (1995). Misuses of statistical analysis in climate
    research. In *Analysis of Climate Variability*, 11-26. Springer.
    Yue, S., Pilon, P., Phinney, B. and Cavadias, G. (2002). The influence of
    autocorrelation on the ability to detect trend in hydrological series.
    *Hydrological Processes*, 16, 1807-1829.

    Examples
    --------
    >>> r = prewhitened_mann_kendall([1.0, 2.1, 2.9, 4.2, 5.1, 5.8, 7.2, 8.1], "pw")
    >>> r.S, round(r.sen_slope, 6)
    (21, 0.382103)
    """
    X = [float(v) for v in x]
    n = len(X)
    if n < 3:
        raise ValueError("need at least three values")
    if method == "pw":
        r1 = _acf1(X)
        y = [X[t + 1] - r1 * X[t] for t in range(n - 1)]
        old = _sen(X)
    elif method == "tfpw":
        old = _sen(X)
        xt = [X[t] - old * (t + 1) for t in range(n)]
        r1 = _acf1(xt)
        y = [xt[t + 1] - r1 * xt[t] + old * (t + 1) for t in range(n - 1)]
    else:
        raise ValueError("method must be 'pw' or 'tfpw'")
    S, var, z = _mk(y)
    m = len(y)
    return RichResult(
        payload={
            "S": S,
            "var_S": var,
            "Z": z,
            "p_value": math.erfc(abs(z) / math.sqrt(2.0)),
            "tau": S / (0.5 * m * (m - 1)),
            "sen_slope": _sen(y),
            "old_sen_slope": old,
            "r1": r1,
        }
    )


def fleiss_kappa(counts) -> RichResult:
    r"""Fleiss' kappa for ``m`` raters, as ``irr::kappam.fleiss``.

    ``counts[i][j]`` is the number of raters assigning subject ``i`` to
    category ``j`` (every row sums to ``m``). ``P_bar = mean_i (sum_j n_ij^2 - m)/(m(m-1))``,
    ``P_e = sum_j p_j^2``, ``kappa = (P_bar - P_e)/(1 - P_e)``; the null
    standard error is Fleiss, Nee and Landis's (1979)
    ``sqrt(2 / (N m (m-1))) sqrt((sum p q)^2 - sum p q (q - p)) / sum p q``.

    References
    ----------
    Fleiss, J. L. (1971). Measuring nominal scale agreement among many
    raters. *Psychological Bulletin*, 76, 378-382.
    Fleiss, J. L., Nee, J. C. M. and Landis, J. R. (1979). Large sample
    variance of kappa in the case of different sets of raters.
    *Psychological Bulletin*, 86, 974-977.

    Examples
    --------
    >>> round(fleiss_kappa([[3, 0], [0, 3], [2, 1], [3, 0]]).kappa, 12)
    0.625
    """
    T = [[float(v) for v in row] for row in counts]
    N, K = len(T), len(T[0])
    m = ssum(T[0])
    if any(abs(ssum(r) - m) > 1e-9 for r in T):
        raise ValueError("every subject needs the same number of ratings")
    pbar = ssum((ssum(v * v for v in r) - m) / (m * (m - 1)) / N for r in T)
    pj = [ssum(T[i][j] for i in range(N)) / (N * m) for j in range(K)]
    pe = ssum(p * p for p in pj)
    kappa = (pbar - pe) / (1 - pe)
    pq = ssum(p * (1 - p) for p in pj)
    var = 2.0 / (pq * pq * (N * m * (m - 1))) * (pq * pq - ssum(p * (1 - p) * ((1 - p) - p) for p in pj))
    se = math.sqrt(var)
    z = kappa / se
    return RichResult(payload={"kappa": kappa, "se": se, "z": z, "p_value": math.erfc(abs(z) / math.sqrt(2.0))})


def empirical_breakdown_point(estimator, x, *, magnitude: float = 1e12, tol: float = 1e6) -> RichResult:
    r"""Empirical finite-sample replacement breakdown point ``epsilon* = m* / n`` of an estimator.

    For ``m = 1, 2, ...`` the ``m`` observations largest in absolute value
    are replaced by ``magnitude * (1 + k)`` (arbitrarily large, distinct);
    breakdown occurs at the smallest ``m`` for which the estimate moves by
    more than ``tol`` times the data's range (Donoho and Huber 1983). The
    sample mean breaks down at ``1/n``, the median at about one half.

    References
    ----------
    Donoho, D. L. and Huber, P. J. (1983). The notion of breakdown point. In
    *A Festschrift for Erich L. Lehmann*, 157-184. Wadsworth.
    Hampel, F. R. (1971). A general qualitative definition of robustness.
    *Annals of Mathematical Statistics*, 42, 1887-1896.

    Examples
    --------
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> empirical_breakdown_point(lambda v: sum(v) / len(v), x).m
    1
    """
    X = [float(v) for v in x]
    n = len(X)
    base = estimator(list(X))
    scale = (max(X) - min(X)) or 1.0
    order = sorted(range(n), key=lambda i: (-abs(X[i]), i))
    for m in range(1, n + 1):
        z = list(X)
        for k, i in enumerate(order[:m]):
            z[i] = magnitude * (1 + k)
        if abs(estimator(z) - base) > tol * scale:
            return RichResult(payload={"m": m, "breakdown_point": m / n})
    return RichResult(payload={"m": n + 1, "breakdown_point": 1.0})


def cheatsheet() -> str:
    return (
        "ace_index / budyko_olr / prewhitened_mann_kendall / fleiss_kappa / empirical_breakdown_point -> "
        "climate and agreement statistics."
    )

# alias kept from the retired placeholder of the same name
breakdown_point = empirical_breakdown_point

# alias kept from the retired placeholder of the same name
outgoing_longwave = budyko_olr

# alias kept from the retired placeholder of the same name
prewhitening_mk = prewhitened_mann_kendall
