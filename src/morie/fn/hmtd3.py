# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Twin delayed DDPG (TD3): two critics + delayed policy updates."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_td3"]


def geron_td3(
    env,
    policy=None,
    Q1=None,
    Q2=None,
    epochs=30,
    lr=0.5,
    gamma=0.9,
    steps=20,
    policy_delay=2,
    tau=0.5,
    noise=0.2,
    seed=0,
):
    """
    Twin delayed DDPG (TD3): two critics + delayed policy updates.

    Formula: target = r + gamma * min(Q1_target, Q2_target)(s', a_tilde)

    All three TD3 tricks are implemented over a finite action set:

    1. **Twin critics.** Both critics regress on the *same* target built
       from ``min(Q1_target, Q2_target)``. Because ``min`` of two noisy
       estimates is biased low, this cancels the positive bias that a
       single critic's ``max`` introduces; the measured gap against the
       single-critic target is returned as `overestimation_gap` and is
       non-negative by construction.
    2. **Target policy smoothing.** The target action ``a_tilde`` is the
       target policy's action perturbed to a neighbouring action with
       probability `noise` and clipped to the action range, so the target
       cannot exploit a sharp spike in Q.
    3. **Delayed policy updates.** The deterministic policy and the target
       networks are refreshed only every `policy_delay` epochs, with
       Polyak averaging at rate `tau`.

    Parameters
    ----------
    env : object
        ``reset() -> s``, ``step(a) -> (s', r, done)``, integer attributes
        ``n_states`` and ``n_actions``.
    policy : array-like, optional
        Initial deterministic policy: one action index per state.
    Q1, Q2 : array-like, optional
        Initial (n_states, n_actions) critics; default zeros.
    epochs, steps : int
        Epochs and environment steps collected per epoch (both >= 1).
    lr : float, default 0.5
        Critic step size in (0, 1].
    gamma : float, default 0.9
        Discount in [0, 1).
    policy_delay : int, default 2
        Epochs between policy / target refreshes (>= 1).
    tau : float, default 0.5
        Polyak rate in (0, 1].
    noise : float, default 0.2
        Probability of perturbing the target action (target smoothing); in [0, 1].
    seed : int, default 0
        LCG seed for exploration and smoothing.

    Returns
    -------
    result : RichResult
        Keys: policy, Q1, Q2, returns, overestimation_gap, policy_updates,
        estimate, n, method.

    Examples
    --------
    A one-state bandit: action 1 pays 1, action 0 pays nothing.

    >>> class Bandit:
    ...     n_states, n_actions = 1, 2
    ...     def reset(self):
    ...         return 0
    ...     def step(self, a):
    ...         return 0, float(a), False
    >>> r = geron_td3(Bandit(), epochs=40)
    >>> int(r["policy"][0])
    1
    >>> bool(r["overestimation_gap"] >= 0.0)
    True
    >>> int(r["policy_updates"])
    20

    References
    ----------
    Géron Ch 19
    """
    for attr in ("reset", "step"):
        if not callable(getattr(env, attr, None)):
            raise ValueError(f"geron_td3: env must provide a callable {attr}()")
    n_s = getattr(env, "n_states", None)
    n_a = getattr(env, "n_actions", None)
    if not isinstance(n_s, (int, np.integer)) or not isinstance(n_a, (int, np.integer)) or n_s < 1 or n_a < 2:
        raise ValueError("geron_td3: env must expose integer n_states >= 1 and n_actions >= 2")
    n_s, n_a = int(n_s), int(n_a)
    E, T = int(epochs), int(steps)
    if E < 1 or T < 1:
        raise ValueError(f"geron_td3: epochs and steps must both be >= 1, got {E} and {T}")
    step_size, g, tau_f, eps = float(lr), float(gamma), float(tau), float(noise)
    if not (0.0 < step_size <= 1.0):
        raise ValueError(f"geron_td3: lr must lie in (0, 1], got {step_size}")
    if not (0.0 <= g < 1.0):
        raise ValueError(f"geron_td3: gamma must lie in [0, 1), got {g}")
    if not (0.0 < tau_f <= 1.0):
        raise ValueError(f"geron_td3: tau must lie in (0, 1], got {tau_f}")
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"geron_td3: noise must lie in [0, 1], got {eps}")
    delay = int(policy_delay)
    if delay < 1:
        raise ValueError(f"geron_td3: policy_delay must be >= 1, got {delay}")

    mu = np.zeros(n_s, dtype=int) if policy is None else np.asarray(policy).ravel().astype(int)
    if mu.size != n_s:
        raise ValueError(f"geron_td3: policy must give one action per state ({n_s}), got {mu.size}")
    if mu.min() < 0 or mu.max() >= n_a:
        raise ValueError(f"geron_td3: policy actions must lie in 0..{n_a - 1}")
    q1 = np.zeros((n_s, n_a)) if Q1 is None else np.asarray(Q1, dtype=float).copy()
    q2 = np.zeros((n_s, n_a)) if Q2 is None else np.asarray(Q2, dtype=float).copy()
    for nm, q in (("Q1", q1), ("Q2", q2)):
        if q.shape != (n_s, n_a):
            raise ValueError(f"geron_td3: {nm} must have shape {(n_s, n_a)}, got {q.shape}")
    q1_t, q2_t, mu_t = q1.copy(), q2.copy(), mu.copy()

    rng = int(seed) % 2**32

    def _u():
        nonlocal rng
        rng = (1664525 * rng + 1013904223) % 2**32
        return (rng + 0.5) / 2**32

    returns = []
    gaps = []
    n_policy_updates = 0
    s = int(env.reset())
    for ep in range(E):
        batch = []
        total = 0.0
        for _ in range(T):
            a = int(_u() * n_a) if _u() < 0.3 else int(mu[s])  # exploration around the deterministic policy
            a = min(max(a, 0), n_a - 1)
            s2, rew, done = env.step(a)
            s2, rew, done = int(s2), float(rew), bool(done)
            if not (0 <= s2 < n_s):
                raise ValueError(f"geron_td3: env.step returned state {s2} outside 0..{n_s - 1}")
            if not np.isfinite(rew):
                raise ValueError("geron_td3: env.step returned a non-finite reward")
            batch.append((s, a, rew, s2, done))
            total += rew
            s = int(env.reset()) if done else s2
        for (bs, ba, br, bs2, bd) in batch:
            a_t = int(mu_t[bs2])
            if _u() < eps:  # target policy smoothing: nudge to a neighbouring action, clipped
                a_t = min(max(a_t + (1 if _u() < 0.5 else -1), 0), n_a - 1)
            twin = min(float(q1_t[bs2, a_t]), float(q2_t[bs2, a_t]))
            single = float(q1_t[bs2, a_t])
            gaps.append(single - twin)
            target = br + (0.0 if bd else g * twin)
            q1[bs, ba] += step_size * (target - q1[bs, ba])
            q2[bs, ba] += step_size * (target - q2[bs, ba])
        if (ep + 1) % delay == 0:
            mu = np.argmax(q1, axis=1).astype(int)
            q1_t = (1 - tau_f) * q1_t + tau_f * q1
            q2_t = (1 - tau_f) * q2_t + tau_f * q2
            mu_t = mu.copy()
            n_policy_updates += 1
        returns.append(total)

    gap = float(np.mean(gaps)) if gaps else 0.0

    return RichResult(
        title="TD3 (twin delayed, tabular)",
        summary_lines=[
            ("States", n_s),
            ("Actions", n_a),
            ("Policy updates", n_policy_updates),
            ("Mean overestimation gap avoided", gap),
            ("Final epoch return", returns[-1]),
        ],
        interpretation=(
            "min(Q1, Q2) is deliberately pessimistic: DDPG's failure mode is a critic that overestimates "
            "and a policy that then chases the error, and the delay stops the policy moving faster than its critic."
        ),
        payload={
            "policy": mu,
            "Q1": q1,
            "Q2": q2,
            "Q1_target": q1_t,
            "Q2_target": q2_t,
            "returns": np.asarray(returns, dtype=float),
            "overestimation_gap": gap,
            "policy_updates": int(n_policy_updates),
            "estimate": float(returns[-1]),
            "n": int(E * T),
            "method": "Tabular TD3: twin critics, clipped target smoothing, delayed Polyak policy/target updates",
        },
    )


def cheatsheet():
    return "hmtd3: Twin delayed DDPG (TD3): two critics + delayed policy updates"
