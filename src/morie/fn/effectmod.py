# morie.fn -- function file (rootcoder007/morie)
"""Discovering effect modification in matched observational studies: Zaykin's truncated product of
P-values and the submax method (maximum of correlated standardized deviates) for Wilcoxon signed-rank
statistics under a sensitivity parameter Gamma."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_normal
from ._s03core import chol

__all__ = [
    "truncated_product_pvalue",
    "wilcoxon_sensitivity_moments",
    "submax_comparisons",
    "submax_test",
]


def truncated_product_pvalue(p, alpha_tilde: float = 0.05) -> RichResult:
    r"""Zaykin et al.'s truncated product ``P^ = prod_{P_l <= alpha~} P_l`` of ``L`` independent P-values and its P-value.

    Written as a binomial mixture of gamma distributions (Rosenbaum's eq. 11.5-11.6):
    ``Pr(P^ <= w) = sum_{k=1}^L C(L, k) a^k (1 - a)^{L-k} (1 - F_k(-log(w / a^k)))``
    with ``F_k`` the gamma(k, 1) distribution function (``F_k(x) = 0`` for
    ``x < 0``); ``P^ = 1`` (P-value 1) when no ``P_l <= alpha~``.

    References
    ----------
    Zaykin, D. V., Zhivotovsky, L. A., Westfall, P. H. and Weir, B. S. (2002).
    Truncated product method for combining P-values. *Genetic Epidemiology*, 22, 170-185.
    Lee, K., Small, D. S. and Rosenbaum, P. R. (2021). Discovering effect
    modification in matched observational studies. In *Handbook of Matching
    and Weighting Adjustments for Causal Inference*, ch. 11. CRC Press.

    Examples
    --------
    >>> round(truncated_product_pvalue([0.025, 0.4]).pvalue, 12)
    0.05
    """
    p = [float(v) for v in p]
    L = len(p)
    a = float(alpha_tilde)
    if not 0 < a <= 1 or L == 0 or min(p) < 0 or max(p) > 1:
        raise ValueError("need 0 < alpha_tilde <= 1 and P-values in [0, 1]")
    w = 1.0
    sel = 0
    for v in p:
        if v <= a:
            w *= v
            sel += 1
    if sel == 0:
        return RichResult(payload={"w": 1.0, "pvalue": 1.0, "n_selected": 0})
    tot = 0.0
    for k in range(1, L + 1):
        x = -math.log(w / a**k) if w > 0 else math.inf
        if x <= 0:
            tail = 1.0
        elif math.isinf(x):
            tail = 0.0
        else:
            term, s = 1.0, 1.0
            for j in range(1, k):
                term *= x / j
                s += term
            tail = math.exp(-x) * s
        tot += math.comb(L, k) * a**k * (1 - a) ** (L - k) * tail
    return RichResult(payload={"w": w, "pvalue": min(tot, 1.0), "n_selected": sel})


def wilcoxon_sensitivity_moments(n_pairs: int, Gamma: float = 1.0) -> RichResult:
    r"""Upper-bound null mean and variance of Wilcoxon's signed-rank statistic from ``n_pairs`` untied pairs at bias ``Gamma``.

    ``mu = Gamma / (1 + Gamma) I (I + 1) / 2`` and
    ``nu = Gamma / (1 + Gamma)^2 I (I + 1)(2I + 1) / 6`` (Rosenbaum 2002, ch. 4).

    Examples
    --------
    >>> r = wilcoxon_sensitivity_moments(10, 1.0)
    >>> r.mu, r.nu
    (27.5, 96.25)
    """
    I = n_pairs  # noqa: E741
    if I < 1 or Gamma < 1:
        raise ValueError("need n_pairs >= 1 and Gamma >= 1")
    k = Gamma / (1 + Gamma)
    return RichResult(
        payload={"mu": k * I * (I + 1) / 2, "nu": Gamma / (1 + Gamma) ** 2 * I * (I + 1) * (2 * I + 1) / 6}
    )


def submax_comparisons(p: int) -> list:
    r"""Comparison matrix of the submax method for ``p`` binary covariates: ``K = 2p + 1`` rows over ``2^p`` groups.

    Row 1 pools all groups; for covariate ``j`` there is one row for the
    groups at level 0 and one for level 1 (group ``g`` has covariate ``j`` at
    level ``(g >> j) & 1``).

    Examples
    --------
    >>> submax_comparisons(1)
    [[1, 1], [1, 0], [0, 1]]
    """
    G = 2**p
    C = [[1] * G]
    for j in range(p):
        C.append([1 if ((g >> j) & 1) == 0 else 0 for g in range(G)])
        C.append([1 if ((g >> j) & 1) == 1 else 0 for g in range(G)])
    return C


def submax_test(T, mu, V, C, alpha: float = 0.05, nsim: int = 100000, seed: int = 1) -> RichResult:
    r"""Submax test of no effect in any of ``K`` overlapping comparisons (Lee, Small and Rosenbaum).

    Group statistics ``T_g`` (independent, null upper-bound means ``mu_g`` and
    variances ``V_g`` at a given Gamma) are combined as ``S = C T``;
    ``D_k = (S_k - theta_k) / sigma_k`` with ``theta = C mu`` and
    ``Sigma = C diag(V) C'``. Under the null ``D`` is approximately
    ``N_K(0, rho)`` with ``rho`` the correlation of ``Sigma``; ``D_max`` is
    compared with the ``1 - alpha`` quantile ``kappa`` of ``max_k Z_k``,
    ``Z ~ N_K(0, rho)``, estimated from ``nsim`` Philox draws (``Z = L e``,
    ``L`` the Cholesky factor of ``rho``).

    References
    ----------
    Lee, K., Small, D. S. and Rosenbaum, P. R. (2018). A powerful approach to
    the study of moderate effect modification in observational studies.
    *Biometrics*, 74, 1161-1170.

    Examples
    --------
    >>> m = wilcoxon_sensitivity_moments(10)
    >>> r = submax_test([40, 30], [m.mu] * 2, [m.nu] * 2, submax_comparisons(1), nsim=20000)
    >>> [round(v, 6) for v in r.D], round(r.kappa, 2)
    ([1.081125, 1.274118, 0.254824], 2.03)
    """
    K, G = len(C), len(T)
    theta = [ssum(C[k][g] * mu[g] for g in range(G)) for k in range(K)]
    S = [ssum(C[k][g] * T[g] for g in range(G)) for k in range(K)]
    Sig = [[ssum(C[k][g] * V[g] * C[m][g] for g in range(G)) for m in range(K)] for k in range(K)]
    sd = [math.sqrt(Sig[k][k]) for k in range(K)]
    D = [(S[k] - theta[k]) / sd[k] for k in range(K)]
    rho = [[Sig[k][m] / (sd[k] * sd[m]) for m in range(K)] for k in range(K)]
    Lc = chol([[rho[k][m] + (1e-12 if k == m else 0.0) for m in range(K)] for k in range(K)])
    e = random_normal(nsim * K, seed=seed)
    mx = []
    for s in range(nsim):
        base = s * K
        best = -math.inf
        for k in range(K):
            z = 0.0
            for j in range(k + 1):
                z += Lc[k][j] * float(e[base + j])
            if z > best:
                best = z
        mx.append(best)
    mx.sort()
    kappa = mx[math.ceil((1 - alpha) * nsim) - 1]
    dmax = max(D)
    return RichResult(payload={"D": D, "rho": rho, "Dmax": dmax, "kappa": kappa, "reject": dmax > kappa})


def cheatsheet() -> str:
    return (
        "truncated_product_pvalue / wilcoxon_sensitivity_moments / submax_comparisons / submax_test -> "
        "effect modification in matched observational studies."
    )
