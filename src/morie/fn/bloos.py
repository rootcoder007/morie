# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Bayesian LOO-CV with Pareto-smoothed importance sampling."""

from __future__ import annotations

import math
from typing import Any


def _lse(v):
    m = max(v)
    if m == -math.inf:
        return -math.inf
    s = 0.0
    for t in v:
        s += math.exp(t - m)
    return m + math.log(s)


def _gpdfit(x):
    """Zhang-Stephens profile-likelihood fit of a generalised Pareto to the
    ascending exceedances ``x``, with the weakly informative prior that
    shrinks k toward 0.5 (as ``posterior::gpdfit``). Returns ``(k, sigma)``."""
    N = len(x)
    M = 30 + int(math.floor(math.sqrt(N)))
    xstar = x[int(math.floor(N / 4.0 + 0.5)) - 1]
    if not xstar > x[0]:
        return math.nan, math.nan
    theta = [1.0 / x[N - 1] + (1.0 - math.sqrt(M / (j - 0.5))) / 3.0 / xstar for j in range(1, M + 1)]
    lt = []
    for a in theta:
        kk = 0.0
        for t in x:
            kk += math.log1p(-a * t)
        kk /= N
        if kk == 0.0 or not math.isfinite(kk) or -a / kk <= 0.0:
            return math.inf, math.nan  # NaN profile likelihood: R's fit gives up the same way
        lt.append(N * (math.log(-a / kk) - kk - 1.0))
    top = _lse(lt)
    th = 0.0
    for a, v in zip(theta, lt):
        th += a * math.exp(v - top)
    k = 0.0
    for t in x:
        k += math.log1p(-th * t)
    k /= N
    sigma = -k / th
    k = (k * N + 0.5 * 10.0) / (N + 10.0)
    if math.isnan(k):
        return math.inf, math.nan
    return k, sigma


def _psis_lw(lr):
    """PSIS on one observation's log importance ratios, as ``loo::psis``.

    The ``M = ceil(min(0.2 S, 3 sqrt(S)))`` largest ratios are replaced by the
    expected order statistics of a generalised Pareto fitted to their
    exceedances (Zhang-Stephens profile fit with the weakly informative
    prior on k), then every log weight is truncated at the largest raw one.
    Returns ``(log_weights, k)``; ``k`` is inf when the tail is too short.
    """
    Sn = len(lr)
    mx = max(lr)
    lw = [t - mx for t in lr]
    M = int(math.ceil(min(0.2 * Sn, 3.0 * math.sqrt(Sn))))
    k = math.inf
    if M >= 5:
        o = sorted(range(Sn), key=lambda i: (lw[i], i))
        tail = o[Sn - M :]
        cut = lw[o[Sn - M - 1]]
        if abs(lw[tail[-1]] - lw[tail[0]]) >= 2.220446049250313e-16 / 100:
            ecut = math.exp(cut)
            x = [math.exp(lw[i]) - ecut for i in tail]
            kk, sigma = _gpdfit(x)
            k = kk
            if math.isfinite(kk):
                for z in range(M):
                    p = (z + 0.5) / M
                    q = sigma * math.expm1(-kk * math.log1p(-p)) / kk if kk != 0.0 else -sigma * math.log1p(-p)
                    lw[tail[z]] = math.log(q + ecut)
    lw = [min(t, 0.0) + mx for t in lw]
    return lw, k


def psis_loo(
    log_lik_matrix,
) -> dict[str, Any]:
    """
    Pareto-smoothed importance sampling LOO-CV (PSIS-LOO).

    For observation ``i`` the importance ratios of the leave-one-out
    posterior are ``r_s = 1 / p(y_i | theta_s)``. Their largest
    ``M = ceil(min(0.2 S, 3 sqrt(S)))`` values are replaced by the expected
    order statistics of a generalised Pareto distribution fitted to the tail
    (Zhang and Stephens 2009 fit, shrunk toward k = 0.5), the weights are
    truncated at the largest raw ratio, and

        elpd_loo_i = log( sum_s w_s p(y_i | theta_s) / sum_s w_s ).

    The fitted shape ``k_hat`` is the reliability diagnostic: above 0.7 the
    importance-sampling estimate for that observation is unreliable.
    ``p_loo = lppd - elpd_loo`` and ``looic = -2 elpd_loo``; ``se`` is
    ``sqrt(n var(elpd_loo_i))``. This reproduces ``loo::loo`` with
    ``r_eff = 1``.

    :param log_lik_matrix: Matrix of log-likelihoods (n_samples, n_obs).
    :return: Dictionary with elpd_loo, p_loo, looic, se, k_hat (list),
        n_high_k (k_hat > 0.7), n_obs, pointwise elpd_loo_i.

    References
    ----------
    Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
        evaluation using leave-one-out cross-validation and WAIC.
        *Statistics and Computing*, 27(5), 1413-1432.
    Vehtari, A., Simpson, D., Gelman, A., Yao, Y. & Gabry, J. (2024). Pareto
        smoothed importance sampling. *JMLR* 25(72), 1-58.
    Zhang, J. and Stephens, M. A. (2009). A new and efficient estimation
        method for the generalized Pareto distribution. *Technometrics* 51,
        316-325.
    """
    L = log_lik_matrix.tolist() if hasattr(log_lik_matrix, "tolist") else [list(r) for r in log_lik_matrix]
    if L and not isinstance(L[0], (list, tuple)):
        L = [L]
    L = [[float(v) for v in r] for r in L]
    n_samples, n_obs = len(L), len(L[0])
    elpd_i, lppd_i, k_hat = [], [], []
    for i in range(n_obs):
        ll = [L[s][i] for s in range(n_samples)]
        lw, k = _psis_lw([-v for v in ll])
        k_hat.append(k)
        elpd_i.append(_lse([a + b for a, b in zip(lw, ll)]) - _lse(lw))
        lppd_i.append(_lse(ll) - math.log(n_samples))
    elpd = 0.0
    lppd = 0.0
    for a, b in zip(elpd_i, lppd_i):
        elpd += a
        lppd += b
    if n_obs > 1:
        m = elpd / n_obs
        v = 0.0
        for a in elpd_i:
            v += (a - m) ** 2
        se = math.sqrt(n_obs * v / (n_obs - 1))
    else:
        se = float("nan")
    return {
        "elpd_loo": elpd,
        "p_loo": lppd - elpd,
        "looic": -2.0 * elpd,
        "se": se,
        "k_hat": k_hat,
        "n_high_k": sum(1 for k in k_hat if not k <= 0.7),
        "n_obs": n_obs,
        "elpd_loo_pointwise": elpd_i,
    }


bloos = psis_loo


def cheatsheet() -> str:
    return "psis_loo({}) -> Bayesian LOO-CV with Pareto-smoothed importance sampling (as loo::loo)."


# compact alias per ledger/NAMING.md
psisloo = psis_loo
