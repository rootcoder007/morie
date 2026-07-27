# morie.fn -- function file (rootcoder007/morie)
"""Cross-method consistency check (g-formula vs IPW vs AIPW)."""

import numpy as np

from ._richresult import RichResult
from .aiptdd import _logit_fit
from .causmrop import causal_robins_g_formula

__all__ = ["g_methods_consistency"]


def g_methods_consistency(y, A, L, tau=0.5):
    r"""Compare three point-treatment ATE estimators; flag divergence.

    Computes the average treatment effect three ways --

    - **g-formula** (outcome model only): standardised OLS means,
    - **IPW** (propensity model only): Hajek-weighted difference with
      weights :math:`A/\hat e + (1-A)/(1-\hat e)`,
    - **AIPW** (both, doubly robust): the Bang-Robins augmented
      estimator,

    and flags ``consistent = False`` when the largest pairwise gap
    exceeds ``tau``. Agreement is a specification check: the three
    rely on different nuisance models, so divergence signals that at
    least one is misspecified (or positivity is failing).

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    A : array-like of {0, 1}, shape (n,)
        Treatment.
    L : array-like, shape (n,) or (n, p)
        Baseline confounders.
    tau : float, default 0.5
        Maximum tolerated pairwise divergence.

    Returns
    -------
    RichResult
        keys: ``ate_gformula``, ``ate_ipw``, ``ate_aipw``,
        ``max_divergence``, ``consistent``, ``tau``, ``n``, ``method``.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in
    mortality studies with a sustained exposure period. *Mathematical
    Modelling*, 7, 1393-1512.

    Bang, H. & Robins, J. M. (2005). Doubly robust estimation in
    missing data and causal inference models. *Biometrics*, 61(4),
    962-972.
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float).ravel()
    L = np.asarray(L, dtype=float)
    if L.ndim == 1:
        L = L[:, None]
    n = y.size
    if A.size != n or L.shape[0] != n:
        raise ValueError("y, A, L must share their first dimension.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("A must be binary 0/1.")
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}.")

    g = causal_robins_g_formula(y, A, L)
    ate_g = g["ate"]

    e = np.clip(_logit_fit(L, A), 0.01, 0.99)
    w1, w0 = A / e, (1 - A) / (1 - e)
    ate_ipw = float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())

    # AIPW with the same outcome model as the g-formula arm
    D = np.column_stack([np.ones(n), A, L, A[:, None] * L])
    b, *_ = np.linalg.lstsq(D, y, rcond=None)

    def m(a):
        a_col = np.full(n, float(a))
        return np.column_stack([np.ones(n), a_col, L, a_col[:, None] * L]) @ b

    m1, m0 = m(1), m(0)
    ate_aipw = float(np.mean(m1 - m0 + A * (y - m1) / e - (1 - A) * (y - m0) / (1 - e)))

    ests = np.array([ate_g, ate_ipw, ate_aipw])
    div = float(ests.max() - ests.min())

    return RichResult(
        payload={
            "ate_gformula": ate_g,
            "ate_ipw": ate_ipw,
            "ate_aipw": ate_aipw,
            "max_divergence": div,
            "consistent": bool(div <= tau),
            "tau": tau,
            "n": int(n),
            "method": "Cross-method consistency check (g-formula vs IPW vs AIPW)",
        }
    )


def cheatsheet():
    return "gmccsm: ATE by g-formula/IPW/AIPW, flag divergence > tau"
