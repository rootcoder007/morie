# morie.fn -- function file (rootcoder007/morie)
"""Closed-form results from Grimmett and Stirzaker's *Probability and Random Processes*: the Erlang
renewal function, Berkson's fallacy correlation, the inverse Mills conditional mean, bivariate-normal
dependence measures, Renyi maximal correlation, the beta-binomial law, AR and telegraph spectra,
Wiener conditional correlation, the simple-epidemic duration, Stirling's ratio and MA autocorrelation."""

from __future__ import annotations

import cmath
import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._s03core import jacobi

__all__ = [
    "erlang_renewal_function",
    "berkson_correlation",
    "mills_conditional_mean",
    "bivariate_normal_dependence",
    "maximal_correlation",
    "beta_binomial_pmf",
    "ar_spectral_density",
    "spectral_density_from_acf",
    "telegraph_process",
    "wiener_conditional_correlation",
    "simple_epidemic_duration",
    "stirling_gamma_ratio",
    "ma_autocorrelation",
]


def erlang_renewal_function(t: float, lam: float, k: int = 2) -> float:
    r"""Renewal function ``m(t) = E N(t)`` for Erlang (gamma ``(lam, k)``, integer ``k``) interarrival times.

    The renewal density of a gamma ``(lam, k)`` process is
    ``u(t) = (lam / k) * sum_{j=0}^{k-1} w_j exp(-lam t (1 - w_j))`` with
    ``w_j = exp(2 pi i j / k)``; integrating gives
    ``m(t) = lam t / k + (1/k) sum_{j>=1} w_j / (1 - w_j) (1 - exp(-lam (1 - w_j) t))``.
    For ``k = 2`` this is Grimmett and Stirzaker's renewal exercise (Section 10.1),
    ``m(t) = lam t / 2 - (1 - exp(-2 lam t)) / 4``.

    References
    ----------
    Grimmett, G. R. and Stirzaker, D. R. (2020). *Probability and Random
    Processes*, 4th edn. Oxford University Press, Section 10.1.

    Examples
    --------
    >>> round(erlang_renewal_function(1.0, 2.0), 12)
    0.754578909722
    """
    if lam <= 0 or k < 1 or t < 0:
        raise ValueError("need lam > 0, integer k >= 1 and t >= 0")
    s = 0.0
    for j in range(1, k):
        w = cmath.exp(2j * math.pi * j / k)
        s += (w / (1 - w) * (1 - cmath.exp(-lam * (1 - w) * t))).real
    return lam * t / k + s / k


def berkson_correlation(gamma: float, a: float, c: float) -> float:
    r"""Correlation between the number in hospital and the number in hospital with disease C (Berkson's fallacy).

    Each member of a group contracts C with probability ``gamma`` and is then
    hospitalised for it with probability ``c``; independently anyone is in
    hospital for another reason with probability ``a``. With ``p = a + c - ac``,
    ``rho = sqrt(gamma p / (1 - gamma p) * (1 - a)(1 - gamma c) / (a + gamma c - a gamma c))``
    (Grimmett and Stirzaker, Problem 3.11.37, Berkson's fallacy); a large value is no evidence of a causal link.

    Examples
    --------
    >>> round(berkson_correlation(0.1, 0.2, 0.5), 12)
    0.449586098066
    """
    for v in (gamma, a, c):
        if not 0 < v < 1:
            raise ValueError("gamma, a and c must lie in (0, 1)")
    p = a + c - a * c
    return math.sqrt(gamma * p / (1 - gamma * p) * (1 - a) * (1 - gamma * c) / (a + gamma * c - a * gamma * c))


def mills_conditional_mean(x: float, rho: float) -> float:
    r"""``E(Y | X > x) = rho * phi(x) / Phi(-x)`` for a standard bivariate normal pair (inverse Mills ratio).

    Grimmett and Stirzaker, Section 4.7 exercises (inverse Mills ratio).

    Examples
    --------
    >>> round(mills_conditional_mean(0.0, 0.5), 12)
    0.398942280401
    """
    if not -1 <= rho <= 1:
        raise ValueError("rho must lie in [-1, 1]")
    phi = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return rho * phi / (0.5 * math.erfc(x / math.sqrt(2.0)))


def bivariate_normal_dependence(rho: float) -> RichResult:
    r"""Mean-square contingency ``phi^2 = rho^2 / (1 - rho^2)`` and mutual information ``-log(1 - rho^2) / 2`` of a bivariate normal pair.

    ``phi^2 = E[f(X, Y) / (g(X) h(Y))] - 1`` so ``1 + phi^2 = 1 / (1 - rho^2)``
    (Grimmett and Stirzaker, Section 4.5 exercises); the mutual information is in nats.

    Examples
    --------
    >>> r = bivariate_normal_dependence(0.6)
    >>> round(r.phi2, 12), round(r.mutual_information, 12)
    (0.5625, 0.223143551314)
    """
    if not -1 < rho < 1:
        raise ValueError("rho must lie in (-1, 1)")
    return RichResult(
        payload={"phi2": rho * rho / (1 - rho * rho), "mutual_information": -0.5 * math.log(1 - rho * rho)}
    )


def maximal_correlation(table) -> RichResult:
    r"""Renyi (Hirschfeld-Gebelein-Renyi) maximal correlation ``m(X, Y) = sup rho(f(X), g(Y))`` of a discrete pair.

    With ``Q_ij = P_ij / sqrt(p_i. p_.j)`` the largest singular value of ``Q``
    is 1 (constant functions) and ``m(X, Y)`` is the second largest; ``m = 0``
    exactly when X and Y are independent (Grimmett and Stirzaker, Chapter 7
    exercises, maximal correlation coefficient). The optimal scores are ``f = u / sqrt(p_i.)`` and
    ``g = Q' f sqrt(p_i.) / (m sqrt(p_.j))``, standardised to unit variance.
    ``table`` holds counts or probabilities (rows X, columns Y).

    References
    ----------
    Renyi, A. (1959). On measures of dependence. *Acta Math. Acad. Sci. Hungar.*, 10, 441-451.

    Examples
    --------
    >>> round(maximal_correlation([[0.3, 0.2], [0.1, 0.4]]).m, 12)
    0.408248290464
    """
    P = [[float(v) for v in row] for row in table]
    tot = ssum(v for row in P for v in row)
    P = [[v / tot for v in row] for row in P]
    r, c = len(P), len(P[0])
    pr = [ssum(row) for row in P]
    pc = [ssum(P[i][j] for i in range(r)) for j in range(c)]
    if min(pr) <= 0 or min(pc) <= 0:
        raise ValueError("every row and column must have positive mass")
    Q = [[P[i][j] / math.sqrt(pr[i] * pc[j]) for j in range(c)] for i in range(r)]
    QQt = [[ssum(Q[i][k] * Q[j][k] for k in range(c)) for j in range(r)] for i in range(r)]
    vals, vecs = jacobi(QQt)
    if r < 2:
        return RichResult(payload={"m": 0.0, "f": [0.0], "g": [0.0] * c})
    m = math.sqrt(max(vals[-2], 0.0))
    u = [vecs[i][r - 2] for i in range(r)]
    f = [u[i] / math.sqrt(pr[i]) for i in range(r)]
    g = [ssum(Q[i][j] * u[i] for i in range(r)) / (m * math.sqrt(pc[j])) if m > 0 else 0.0 for j in range(c)]
    return RichResult(payload={"m": m, "f": f, "g": g})


def beta_binomial_pmf(k: int, n: int, a: float, b: float) -> float:
    r"""Beta-binomial (negative hypergeometric) probability ``p_k = C(n, k) B(a + k, n + b - k) / B(a, b)``.

    Grimmett and Stirzaker (Polya urn, negative hypergeometric distribution): ``p_k`` is proportional to
    ``C(n, k) Gamma(a + k) Gamma(n + b - k)``, ``k = 0, ..., n``.

    Examples
    --------
    >>> round(beta_binomial_pmf(2, 5, 2.0, 3.0), 12)
    0.238095238095
    """
    if a <= 0 or b <= 0 or not 0 <= k <= n:
        raise ValueError("need a, b > 0 and 0 <= k <= n")
    lg = math.lgamma
    return math.exp(
        lg(n + 1) - lg(k + 1) - lg(n - k + 1) + lg(a + k) + lg(n + b - k) - lg(n + a + b) + lg(a + b) - lg(a) - lg(b)
    )


def _ar_unnormalised(lam, alpha, sigma2):
    z = complex(1.0, 0.0)
    for j, a in enumerate(alpha, start=1):
        z -= a * cmath.exp(-1j * j * lam)
    return sigma2 / (2 * math.pi * (z.real * z.real + z.imag * z.imag))


def ar_spectral_density(lam: float, alpha, sigma2: float = 1.0, normalized: bool = True, ngrid: int = 4096) -> float:
    r"""Spectral density of the stationary autoregression ``Y_n = sum_j alpha_j Y_{n-j} + Z_n``.

    Unnormalised, ``f(lam) = sigma2 / (2 pi |1 - sum_j alpha_j e^{-i j lam}|^2)`` on
    ``[-pi, pi]``. Grimmett and Stirzaker (Example 9.3.23) normalise by the
    variance ``c(0) = int f``, so that the density is that of the spectral
    distribution; for AR(1) this is ``(1 - a^2) / (2 pi (1 - 2 a cos lam + a^2))``.
    ``c(0)`` is the periodic trapezoidal rule on ``ngrid`` points (spectrally accurate).

    Examples
    --------
    >>> round(ar_spectral_density(0.5, [0.4]), 12)
    0.291941997432
    """
    alpha = [float(v) for v in alpha]
    f = _ar_unnormalised(lam, alpha, sigma2)
    if not normalized:
        return f
    h = 2 * math.pi / ngrid
    c0 = ssum(_ar_unnormalised(-math.pi + i * h, alpha, sigma2) for i in range(ngrid)) * h
    return f / c0


def spectral_density_from_acf(lam: float, rho) -> float:
    r"""``f(lam) = (1 / 2 pi) sum_n rho(n) cos(n lam)`` for a real stationary sequence (Grimmett and Stirzaker 9.3.17).

    ``rho = [rho(1), rho(2), ...]`` is the autocorrelation at positive lags
    (``rho(0) = 1``); the series is summed over the lags supplied.

    Examples
    --------
    >>> round(spectral_density_from_acf(0.0, [0.5]), 12)
    0.318309886184
    """
    s = 1.0
    for n, r in enumerate(rho, start=1):
        s += 2 * float(r) * math.cos(n * lam)
    return s / (2 * math.pi)


def telegraph_process(t: float, lam: float, alpha: float, beta: float) -> RichResult:
    r"""Two-state Markov (telegraph) process with rates ``alpha`` and ``beta``: autocorrelation and spectral density.

    ``rho(t) = exp(-|t| (alpha + beta))`` and the spectral density is Cauchy,
    ``f(lam) = (alpha + beta) / (pi ((alpha + beta)^2 + lam^2))`` (Grimmett and Stirzaker, Example 9.3.21).

    Examples
    --------
    >>> r = telegraph_process(0.5, 1.0, 1.0, 2.0)
    >>> round(r.rho, 12), round(r.f, 12)
    (0.223130160148, 0.095492965855)
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("rates must be positive")
    s = alpha + beta
    return RichResult(payload={"rho": math.exp(-abs(t) * s), "f": s / (math.pi * (s * s + lam * lam))})


def wiener_conditional_correlation(s: float, t: float, u: float, v: float) -> float:
    r"""Correlation of ``W(t)`` and ``W(u)`` given ``W(s)`` and ``W(v)``, ``s < t < u < v``: ``sqrt((v - u)(t - s) / ((v - t)(u - s)))``.

    Grimmett and Stirzaker, Section 8.5 exercises (Brownian-bridge covariance).

    Examples
    --------
    >>> round(wiener_conditional_correlation(0, 1, 2, 4), 12)
    0.57735026919
    """
    if not s < t < u < v:
        raise ValueError("need s < t < u < v")
    return math.sqrt((v - u) * (t - s) / ((v - t) * (u - s)))


def simple_epidemic_duration(N: int, lam: float) -> RichResult:
    r"""Time ``T`` for a simple epidemic (birth process, rates ``lam i (N + 1 - i)``, ``X(0) = 1``) to infect all ``N + 1``.

    ``T`` is a sum of independent exponentials, so
    ``E T = sum_i 1 / (lam i (N + 1 - i))`` and ``var T = sum_i 1 / (lam i (N + 1 - i))^2``
    (Grimmett and Stirzaker, Section 6.12, a non-linear epidemic).

    Examples
    --------
    >>> round(simple_epidemic_duration(3, 1.0).mean, 12)
    0.916666666667
    """
    if N < 1 or lam <= 0:
        raise ValueError("need N >= 1 and lam > 0")
    rates = [lam * i * (N + 1 - i) for i in range(1, N + 1)]
    return RichResult(payload={"mean": ssum(1 / r for r in rates), "variance": ssum(1 / (r * r) for r in rates)})


def stirling_gamma_ratio(t: float) -> float:
    r"""``Gamma(t) / (t^{t - 1/2} e^{-t} sqrt(2 pi))``, which tends to 1 as ``t -> inf`` (Stirling's formula).

    Grimmett and Stirzaker, Section 5.9 and Problem 7.11.38.

    Examples
    --------
    >>> round(stirling_gamma_ratio(10.0), 12)
    1.008365359132
    """
    if t <= 0:
        raise ValueError("t must be positive")
    return math.exp(math.lgamma(t) - (t - 0.5) * math.log(t) + t - 0.5 * math.log(2 * math.pi))


def ma_autocorrelation(theta, max_lag: int) -> list:
    r"""Autocorrelation ``rho(0..max_lag)`` of ``Y_n = Z_n + sum_j theta_j Z_{n-j}``.

    ``rho(k) = sum_{j=0}^{q-k} theta_j theta_{j+k} / sum_j theta_j^2`` with
    ``theta_0 = 1`` and zero beyond lag ``q``; for MA(1),
    ``rho(1) = a / (1 + a^2)`` (Grimmett and Stirzaker, Section 9.1 exercises).

    Examples
    --------
    >>> [round(v, 12) for v in ma_autocorrelation([0.5], 2)]
    [1.0, 0.4, 0.0]
    """
    th = [1.0] + [float(v) for v in theta]
    q = len(th) - 1
    den = ssum(v * v for v in th)
    out = []
    for k in range(max_lag + 1):
        out.append(ssum(th[j] * th[j + k] for j in range(q - k + 1)) / den if k <= q else 0.0)
    return out


def cheatsheet() -> str:
    return (
        "erlang_renewal_function / berkson_correlation / mills_conditional_mean / bivariate_normal_dependence / "
        "maximal_correlation / beta_binomial_pmf / ar_spectral_density / spectral_density_from_acf / "
        "telegraph_process / wiener_conditional_correlation / simple_epidemic_duration / stirling_gamma_ratio / "
        "ma_autocorrelation -> closed-form results from Grimmett and Stirzaker."
    )
