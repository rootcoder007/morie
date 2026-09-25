# morie.fn -- function file (rootcoder007/morie)
"""G-computation for time-varying treatments.

Extends the parametric g-formula to the time-varying setting by
sequentially fitting outcome models at each time point and computing
the counterfactual outcome under a specified treatment regime.

References
----------
Robins, J. M. (1986). A new approach to causal inference in
mortality studies with a sustained exposure period.
*Mathematical Modelling*, 7(9-12), 1393-1512.

Hernan, M. A., & Robins, J. M. (2020). *Causal Inference: What If*.
Chapman & Hall/CRC. Part III.

Daniel, R. M., Cousens, S. N., De Stavola, B. L., Kenward, M. G.,
& Sterne, J. A. C. (2013). Methods for dealing with time-dependent
confounding. *Statistics in Medicine*, 32(9), 1584-1618.
"""

from __future__ import annotations

from typing import Any

from . import _array_core as np
from ._stats_core import norm as _norm

__all__ = ["gcomp"]


def gcomp(
    Y: np.ndarray,
    T_seq: np.ndarray,
    L_seq: np.ndarray,
    *,
    regime: str = "always_treat",
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    r"""G-computation for time-varying treatment regimes.

    The iterated-conditional-expectation form of Robins' g-formula
    (Bang and Robins 2005; Hernan and Robins 2020, ch. 21). For a static
    regime :math:`\bar a^*`, set :math:`Q_{K+1} = Y` and for
    :math:`k = K, \ldots, 1` regress :math:`Q_{k+1}` on the observed
    history :math:`(\bar A_k, \bar L_k)` by least squares, then predict
    with :math:`\bar A_k = \bar a^*_k` and the observed
    :math:`\bar L_k`; the mean of :math:`Q_1` estimates
    :math:`E[Y(\bar a^*)]`.

    The previous version regressed Y on the whole history once and swapped
    the treatments in, keeping the observed covariates. Under
    treatment-confounder feedback (A_1 -> L_2 -> Y) that holds L_2 at its
    observed value and drops the part of A_1's effect that runs through
    it; the backward regressions average L_2 over its distribution given
    the earlier history, which is what the g-formula requires.

    Parameters
    ----------
    Y : np.ndarray
        Final outcome, shape ``(n,)``.
    T_seq : np.ndarray
        Treatment matrix, shape ``(n, K)``; one column per time point.
    L_seq : np.ndarray
        Time-varying covariate matrix, shape ``(n, K, p)`` or
        ``(n, K)`` for a single covariate per time point.
    regime : str
        ``"always_treat"`` (:math:`\bar{A}^*=1`),
        ``"never_treat"`` (:math:`\bar{A}^*=0`), or
        ``"natural"`` (observed treatment).
    alpha : float
        Significance level.
    n_boot : int
        Bootstrap replicates for standard error.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``ate`` (difference always vs never), ``mean_treated``,
        ``mean_control``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.

    References
    ----------
    Robins (1986). Mathematical Modelling, 7(9-12), 1393-1512.
    Daniel et al. (2013). Statistics in Medicine, 32(9), 1584-1618.
    """
    Y = np.asarray(Y, dtype=float)
    T_seq = np.asarray(T_seq, dtype=float)
    if T_seq.ndim == 1:
        T_seq = T_seq[:, None]
    L_seq = np.asarray(L_seq, dtype=float)
    if L_seq.ndim == 2:
        L_seq = L_seq[:, :, None]

    n, K = T_seq.shape
    if len(Y) != n:
        raise ValueError("Y and T_seq must have the same number of rows.")
    if L_seq.shape[0] != n or L_seq.shape[1] != K:
        raise ValueError("L_seq shape must be (n, K) or (n, K, p).")

    rng = np.random.default_rng(seed)
    yl = [float(v) for v in Y.tolist()]
    tl = [[float(v) for v in row] for row in T_seq.tolist()]
    ll = L_seq.tolist()
    p = len(ll[0][0])

    def _ols_fit_predict(rows, target, rows_new):
        X = np.array(rows)
        b = np.linalg.lstsq(X, np.array(target), rcond=None)[0]
        bl = [float(v) for v in b.tolist()]
        return [sum(c * x for c, x in zip(bl, r)) for r in rows_new]

    def _estimate(idx):
        idx = [int(i) for i in (idx.tolist() if hasattr(idx, "tolist") else idx)]
        y_ = [yl[i] for i in idx]
        if regime == "natural":
            return sum(y_) / len(y_), None, None

        def _mean_under(astar):
            q = list(y_)
            for k in range(K - 1, -1, -1):
                hist_obs, hist_reg = [], []
                for i in idx:
                    lk = [ll[i][j][c] for j in range(k + 1) for c in range(p)]
                    hist_obs.append([1.0] + tl[i][: k + 1] + lk)
                    hist_reg.append([1.0] + [astar] * (k + 1) + lk)
                q = _ols_fit_predict(hist_obs, q, hist_reg)
            return sum(q) / len(q)

        mu1 = _mean_under(1.0)
        mu0 = _mean_under(0.0)
        return mu1 - mu0, mu1, mu0

    ate_obs, mu1_obs, mu0_obs = _estimate(np.arange(n))

    # Bootstrap SE
    boot_ates = np.empty(n_boot)
    for b in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        boot_ates[b] = _estimate(boot_idx)[0]
    se = float(np.std(boot_ates, ddof=1))
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "ate": float(ate_obs),
        "mean_treated": float(mu1_obs) if mu1_obs is not None else None,
        "mean_control": float(mu0_obs) if mu0_obs is not None else None,
        "se": se,
        "ci_lower": float(ate_obs) - z * se,
        "ci_upper": float(ate_obs) + z * se,
        "n": n,
        "method": "g-computation",
    }


def cheatsheet() -> str:
    return "gcomp(Y, T_seq, L_seq) -> G-computation for time-varying treatments (Robins 1986)."
