# morie.fn -- function file (rootcoder007/morie)
"""Generalised advantage estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["generalized_advantage_estimation"]


def generalized_advantage_estimation(rewards, values, gamma=0.99, lam=0.95,
                                     dones=None, last_value=0.0,
                                     normalize=False):
    r"""GAE(:math:`\gamma`, :math:`\lambda`) advantages.

    .. math::
       \delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t),
       \qquad
       \hat A_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}

    :math:`\lambda` interpolates between the two extremes of the
    bias-variance trade-off and does so smoothly. At
    :math:`\lambda = 0` the advantage is the one-step TD residual:
    minimum variance, maximum bias, since it trusts the value function
    completely. At :math:`\lambda = 1` it is the Monte-Carlo return
    minus the baseline: unbiased, but carrying the variance of the
    whole trajectory. The useful values sit near 0.95 because a
    slightly biased advantage with far less variance gives a better
    gradient than an unbiased one that is mostly noise.

    ``effective_horizon`` :math:`= 1/(1-\gamma\lambda)` is how many
    steps of reward actually influence each advantage -- the number to
    look at when deciding whether the rollout length is long enough for
    the chosen parameters. A rollout much shorter than this truncates
    the sum and reintroduces bias that the :math:`\lambda` choice was
    meant to control.

    Parameters
    ----------
    rewards : array-like, shape (T,)
    values : array-like, shape (T,) or (T+1,)
    gamma, lam : float
    dones : array-like of {0, 1}, shape (T,), optional
    last_value : float
        Bootstrap value past the end; ignored if ``values`` has T+1
        entries.
    normalize : bool
        Standardise the advantages, as PPO implementations do.

    Returns
    -------
    RichResult
        ``advantages``, ``returns``, ``td_errors``,
        ``effective_horizon``, ``truncation_bias``.

    References
    ----------
    Schulman, Moritz, Levine, Jordan and Abbeel (2016),
    "High-dimensional continuous control using generalized advantage
    estimation", ICLR, arXiv:1506.02438.

    Examples
    --------
    >>> out = generalized_advantage_estimation([1.0, 1.0], [0.0, 0.0],
    ...                                        gamma=1.0, lam=1.0)
    >>> [round(float(a), 4) for a in out["advantages"]]
    [2.0, 1.0]
    """
    r = np.asarray(rewards, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    T = r.size
    if T < 1:
        raise ValueError("need at least one step.")
    if v.size == T + 1:
        vt, boot = v[:T], float(v[T])
    elif v.size == T:
        vt, boot = v, float(last_value)
    else:
        raise ValueError(
            "values must have T or T+1 entries, got %d for T = %d."
            % (v.size, T)
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1], got %r." % gamma)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1], got %r." % lam)
    d = np.zeros(T) if dones is None else np.asarray(
        dones, dtype=float
    ).ravel()
    if d.size != T:
        raise ValueError("dones has %d entries for %d steps." % (d.size, T))
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise ValueError("dones must be binary 0/1.")

    nxt = np.concatenate([vt[1:], [boot]])
    delta = r + gamma * nxt * (1.0 - d) - vt
    adv = np.zeros(T)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = delta[t] + gamma * lam * (1.0 - d[t]) * acc
        adv[t] = acc
    ret = adv + vt
    raw = adv.copy()
    if normalize:
        s = float(adv.std())
        adv = (adv - adv.mean()) / (s if s > 0 else 1.0)

    gl = gamma * lam
    horizon = float(1.0 / (1.0 - gl)) if gl < 1 else np.inf
    return RichResult(
        payload={
            "estimate": adv,
            "advantages": adv,
            "advantages_raw": raw,
            "returns": ret,
            "td_errors": delta,
            "gamma": float(gamma),
            "lam": float(lam),
            "effective_horizon": horizon,
            "horizon_note": (
                "1/(1 - gamma lambda) steps of reward actually influence "
                "each advantage; a rollout much shorter than this truncates "
                "the sum and puts back the bias lambda was chosen to control"
            ),
            "truncation_bias": bool(np.isfinite(horizon) and horizon > T),
            "tradeoff_note": (
                "lambda = 0 is the one-step TD residual, least variance and "
                "most bias; lambda = 1 is Monte Carlo, unbiased with the "
                "variance of the whole trajectory"
            ),
            "normalized": bool(normalize),
            "mean_advantage": float(raw.mean()),
            "n_steps": int(T),
            "method": "Generalized advantage estimation",
        }
    )


def cheatsheet():
    return (
        "gae: GAE(gamma, lambda) advantages with the effective horizon and "
        "a rollout-truncation warning"
    )
