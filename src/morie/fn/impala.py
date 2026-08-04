# morie.fn -- function file (rootcoder007/morie)
"""IMPALA V-trace off-policy correction."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["impala_vtrace"]


def impala_vtrace(rewards, values, behavior_logp, target_logp, gamma=0.99,
                  rho_bar=1.0, c_bar=1.0, bootstrap_value=0.0):
    r"""V-trace targets for off-policy actor-critic learning.

    .. math::
        v_t = V(x_t) + \sum_{s=t}^{T-1} \gamma^{s-t}
            \left(\prod_{i=t}^{s-1} c_i\right)
            \delta_s V,
        \qquad \delta_s V = \rho_s\left(r_s + \gamma V(x_{s+1}) - V(x_s)\right),

    with truncated importance weights
    :math:`\rho_s = \min(\bar\rho, \pi/\mu)` and
    :math:`c_i = \min(\bar c, \pi/\mu)`.

    The two truncation levels do different jobs and are routinely conflated.
    :math:`\bar\rho` controls **what is learned**: the fixed point is the
    value function of a policy between the behaviour and target policies, and
    :math:`\bar\rho = \infty` recovers the target policy exactly while
    :math:`\bar\rho = 0` recovers the behaviour policy. :math:`\bar c`
    controls **how fast**: it bounds the variance of the trace product, and
    affects the contraction rate but not the fixed point.

    Untruncated importance sampling over a long trajectory has variance that
    explodes multiplicatively -- the product of many ratios -- which is what
    makes naive off-policy correction unusable and V-trace necessary.

    Parameters
    ----------
    rewards : array-like
        Rewards ``(T,)``.
    values : array-like
        Value estimates ``(T,)``.
    behavior_logp, target_logp : array-like
        Log-probabilities of the taken actions under each policy.
    gamma : float
        Discount in [0, 1].
    rho_bar, c_bar : float
        Truncation levels.
    bootstrap_value : float
        Value beyond the horizon.

    Returns
    -------
    RichResult
        ``vs`` (V-trace targets), ``advantage``, ``rho``, ``c``,
        ``n_truncated_rho``.

    References
    ----------
    Espeholt, L., Soyer, H., Munos, R., et al. (2018). IMPALA: Scalable
        distributed deep-RL with importance weighted actor-learner
        architectures. *ICML 2018*, 1407-1416.

    Examples
    --------
    On-policy data leaves the importance weights at 1, so V-trace reduces to
    the usual n-step return.

    >>> import numpy as np
    >>> r = np.ones(5); v = np.zeros(5); lp = np.full(5, -0.7)
    >>> out = impala_vtrace(r, v, lp, lp, gamma=0.9)
    >>> bool(np.all(out["rho"] == 1.0))
    True
    >>> bool(abs(out["vs"][-1] - 1.0) < 1e-9)
    True

    Off-policy data truncates the weights, which is what bounds the variance.

    >>> off = impala_vtrace(r, v, np.full(5, -3.0), np.full(5, -0.1), gamma=0.9)
    >>> bool(off["rho"].max() <= 1.0 and int(off["n_truncated_rho"]) == 5)
    True

    A larger rho_bar moves the fixed point toward the target policy, giving
    different targets.

    >>> wide = impala_vtrace(r, v, np.full(5, -3.0), np.full(5, -0.1),
    ...                      gamma=0.9, rho_bar=5.0)
    >>> bool(wide["vs"][0] > off["vs"][0])
    True
    """
    rw = np.atleast_1d(np.asarray(rewards, dtype=float)).ravel()
    V = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    blp = np.atleast_1d(np.asarray(behavior_logp, dtype=float)).ravel()
    tlp = np.atleast_1d(np.asarray(target_logp, dtype=float)).ravel()
    T = rw.size
    if not (V.size == blp.size == tlp.size == T):
        raise ValueError("rewards, values and both log-probability arrays must agree")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    ratio = np.exp(np.clip(tlp - blp, -50, 50))
    rho = np.minimum(rho_bar, ratio)
    c = np.minimum(c_bar, ratio)
    V_next = np.r_[V[1:], bootstrap_value]
    delta = rho * (rw + gamma * V_next - V)
    vs_minus = np.zeros(T)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = delta[t] + gamma * c[t] * acc
        vs_minus[t] = acc
    vs = V + vs_minus
    vs_next = np.r_[vs[1:], bootstrap_value]
    adv = rho * (rw + gamma * vs_next - V)
    return RichResult(
        title="V-trace targets",
        summary_lines=[("T", int(T)), ("rho_bar", float(rho_bar)),
                       ("c_bar", float(c_bar)),
                       ("truncated rho", int(np.sum(ratio > rho_bar)))],
        warnings=["rho_bar sets WHAT is learned (the fixed point lies between "
                  "behaviour and target policy); c_bar sets HOW FAST, and does "
                  "not move the fixed point"],
        payload={
            "vs": vs, "advantage": adv, "rho": rho, "c": c,
            "delta": delta, "ratio": ratio,
            "n_truncated_rho": int(np.sum(ratio > rho_bar)),
            "n_truncated_c": int(np.sum(ratio > c_bar)),
            "method": "impala_vtrace",
        },
    )


def cheatsheet():
    return "impala: rho_bar sets the FIXED POINT, c_bar sets the RATE -- not interchangeable"


# compact alias per ledger/NAMING.md
impalavtrace = impala_vtrace
