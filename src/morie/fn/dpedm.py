# morie.fn -- function file (rootcoder007/morie)
"""Empirical (epsilon, delta) check for an approximate-DP mechanism."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["approx_dp"]


def approx_dp(mech, D, D_prime, epsilon=1.0, n_samples=20000, bins=50, seed=None):
    r"""Estimate the :math:`\delta` a mechanism needs at a given :math:`\varepsilon`.

    Approximate DP relaxes the pure definition to

    .. math::
        P(M(D) \in S) \le e^{\varepsilon} P(M(D') \in S) + \delta,

    so :math:`\delta` is the probability mass on which the
    :math:`e^{\varepsilon}` bound simply fails. Estimated here as the
    *privacy-loss tail*: the mass of :math:`M(D)` in the region where the
    likelihood ratio exceeds :math:`e^{\varepsilon}`.

    :math:`\delta` is not a rounding error. A mechanism satisfying
    :math:`(\varepsilon, \delta)`-DP may publish the entire database with
    probability :math:`\delta` and remain compliant, which is why
    :math:`\delta \ll 1/n` is the standard requirement -- at
    :math:`\delta = 1/n` a per-record leak is entirely permissible.

    Same caveat as :func:`~morie.fn.dpepsm.epsilon_dp`: a lower bound from one
    dataset pair, useful for catching a broken implementation, useless as a
    certificate.

    Parameters
    ----------
    mech : callable
        ``mech(D, rng) -> float``.
    D, D_prime : array-like
        Neighbouring datasets.
    epsilon : float
        The epsilon at which to measure delta.
    n_samples, bins : int
        Sampling and binning controls.
    seed : int, optional
        Seed.

    Returns
    -------
    RichResult
        ``delta_empirical``, ``epsilon``, ``n_violating_bins``.

    References
    ----------
    Dwork, C., & Roth, A. (2014). The algorithmic foundations of
        differential privacy. *FnT-TCS*, 9(3-4), 211-407.

    Examples
    --------
    A Gaussian mechanism generously calibrated needs almost no delta at its
    nominal epsilon.

    >>> import numpy as np
    >>> D, Dp = np.zeros(10), np.r_[np.zeros(9), 1.0]
    >>> gauss = lambda X, rng: float(X.sum() + rng.normal(0, 3.0))
    >>> r = approx_dp(gauss, D, Dp, epsilon=1.0, n_samples=30000, seed=0)
    >>> bool(r["delta_empirical"] < 0.05)
    True

    A badly under-noised mechanism needs a large delta -- it cannot meet the
    epsilon bound on a substantial share of its output range.

    >>> weak = lambda X, rng: float(X.sum() + rng.normal(0, 0.05))
    >>> bool(approx_dp(weak, D, Dp, epsilon=1.0, n_samples=20000,
    ...                seed=0)["delta_empirical"] > 0.2)
    True

    Raising epsilon lowers the delta required, as the definition implies.

    >>> a = approx_dp(weak, D, Dp, epsilon=0.5, n_samples=20000, seed=0)
    >>> b = approx_dp(weak, D, Dp, epsilon=8.0, n_samples=20000, seed=0)
    >>> bool(b["delta_empirical"] <= a["delta_empirical"])
    True
    """
    rng = np.random.default_rng(seed)
    a = np.asarray([np.ravel(mech(np.asarray(D, dtype=float), rng))[0]
                    for _ in range(int(n_samples))], dtype=float)
    b = np.asarray([np.ravel(mech(np.asarray(D_prime, dtype=float), rng))[0]
                    for _ in range(int(n_samples))], dtype=float)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return RichResult(
            title="Empirical delta",
            summary_lines=[("delta (empirical)", 1.0)],
            warnings=["the mechanism is deterministic; it provides no privacy"],
            payload={"delta_empirical": 1.0, "epsilon": float(epsilon),
                     "n_violating_bins": 0, "method": "approx_dp"},
        )
    edges = np.linspace(lo, hi, int(bins) + 1)
    ca, _ = np.histogram(a, bins=edges)
    cb, _ = np.histogram(b, bins=edges)
    pa = ca / max(ca.sum(), 1)
    pb = cb / max(cb.sum(), 1)
    # delta is the mass of M(D) where the e^eps bound fails outright.
    viol = pa > np.exp(epsilon) * pb
    delta = float(np.sum(np.maximum(pa[viol] - np.exp(epsilon) * pb[viol], 0.0)))
    return RichResult(
        title="Empirical delta",
        summary_lines=[("epsilon", float(epsilon)), ("delta (empirical)", delta),
                       ("violating bins", int(viol.sum()))],
        warnings=["a lower bound from one dataset pair; and note that delta is "
                  "not slack -- at delta = 1/n a per-record leak is permitted"],
        payload={
            "delta_empirical": delta, "epsilon": float(epsilon),
            "n_violating_bins": int(viol.sum()),
            "n_samples": int(n_samples), "method": "approx_dp",
        },
    )


def cheatsheet():
    return "dpedm: delta is the mass where the e^eps bound FAILS; keep delta << 1/n or a record can leak"
