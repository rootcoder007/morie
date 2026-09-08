# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deep deterministic policy gradient (DDPG)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_ddpg"]


def geron_ddpg(
    env,
    actor,
    critic,
    epochs=20,
    lr=0.01,
    gamma=0.95,
    tau=0.01,
    ou_theta=0.15,
    ou_sigma=0.2,
    seed=0,
    s0=None,
    actor_target=None,
    critic_target=None,
):
    """
    Deep deterministic policy gradient (DDPG).

    Formula: off-policy actor-critic with deterministic policy and
    Ornstein-Uhlenbeck exploration

    A working DDPG with linear function approximators, which keeps every
    gradient exact and checkable:

    * actor ``mu(s) = actor . s``, deterministic (no distribution);
    * critic ``Q(s,a) = critic . [s, a]``;
    * critic update: TD error against the *target* nets,
      ``r + gamma Q'(s', mu'(s')) - Q(s,a)``;
    * actor update: the deterministic policy gradient
      ``dQ/da * dmu/dtheta``, which for these linear models is
      ``w_a * s`` -- the chain rule Silver's theorem licenses;
    * target nets tracked by Polyak averaging ``w' <- tau w + (1-tau) w'``.

    Exploration is an Ornstein-Uhlenbeck process,
    ``n <- n + theta*(-n) + sigma*z``, which is temporally correlated
    rather than white -- the point being that a deterministic policy in a
    physical system needs momentum in its noise to explore at all. The
    driving noise is a deterministic LCG/Box-Muller stream, so runs
    reproduce.

    ``env`` must be a callable ``env(s, a) -> (s2, r, done)``.

    Parameters
    ----------
    env : callable
    actor : array-like, shape (d,)
        Initial actor weights (state dimension ``d``).
    critic : array-like, shape (d + 1,)
        Initial critic weights over ``[s, a]``.
    epochs : int, default 20
    lr : float, default 0.01
    gamma : float, default 0.95
    tau : float, default 0.01
        Polyak coefficient in (0, 1].
    ou_theta, ou_sigma : float
        OU mean-reversion and noise scale.
    seed : int, default 0
    s0 : array-like, optional
        Initial state; default all ones.
    actor_target, critic_target : array-like, optional
        Initial target-net weights; default copies of ``actor``/``critic``.

    Returns
    -------
    result : RichResult
        Keys: actor, critic, actor_target, critic_target, critic_losses,
        rewards, actions, ou_noise, q_values, estimate, n, method.

    Examples
    --------
    A one-dimensional environment whose reward is ``-(a - 1)^2``: the
    critic learns and the mean reward improves over training.

    >>> def env(s, a):
    ...     return s, -float((a - 1.0) ** 2), False
    >>> r = geron_ddpg(env, [0.0], [0.0, 0.0], epochs=200, lr=0.05, ou_sigma=0.1, seed=1)
    >>> sum(r["rewards"][-50:]) > sum(r["rewards"][:50])
    True
    >>> len(r["rewards"]), len(r["critic_losses"])
    (200, 200)

    With zero noise the policy is exactly deterministic, so the action is
    the actor's linear output:

    >>> r2 = geron_ddpg(env, [0.5], [0.0, 0.0], epochs=1, lr=0.0, ou_sigma=0.0, s0=[2.0])
    >>> round(r2["actions"][0], 12)
    1.0
    >>> r2["ou_noise"][0]
    0.0

    Polyak averaging moves the target only a fraction of the way:

    >>> r3 = geron_ddpg(env, [0.0], [1.0, 1.0], epochs=1, lr=0.0, tau=0.25, ou_sigma=0.0,
    ...                 critic_target=[0.0, 0.0])
    >>> [round(v, 6) for v in r3["critic_target"]]
    [0.25, 0.25]

    References
    ----------
    Géron Ch 19
    """
    if not callable(env):
        raise ValueError("geron_ddpg: env must be a callable env(s, a) -> (s2, r, done)")
    th = np.atleast_1d(np.asarray(actor, dtype=float)).copy()
    w = np.atleast_1d(np.asarray(critic, dtype=float)).copy()
    if th.size == 0:
        raise ValueError("geron_ddpg: actor is empty")
    if w.size != th.size + 1:
        raise ValueError(
            f"geron_ddpg: critic must have {th.size + 1} weights (state dim {th.size} plus the action), got {w.size}"
        )
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_ddpg: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if eta < 0:
        raise ValueError(f"geron_ddpg: lr must be non-negative, got {lr!r}")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_ddpg: gamma must lie in [0, 1], got {gamma!r}")
    t = float(tau)
    if not (0.0 < t <= 1.0):
        raise ValueError(f"geron_ddpg: tau must lie in (0, 1], got {tau!r}")
    s = np.ones(th.size) if s0 is None else np.atleast_1d(np.asarray(s0, dtype=float)).copy()
    if s.size != th.size:
        raise ValueError(f"geron_ddpg: s0 has {s.size} entries but the actor expects {th.size}")

    th_t = th.copy() if actor_target is None else np.atleast_1d(np.asarray(actor_target, dtype=float)).copy()
    w_t = w.copy() if critic_target is None else np.atleast_1d(np.asarray(critic_target, dtype=float)).copy()
    if th_t.shape != th.shape:
        raise ValueError(f"geron_ddpg: actor_target has shape {th_t.shape} but actor has shape {th.shape}")
    if w_t.shape != w.shape:
        raise ValueError(f"geron_ddpg: critic_target has shape {w_t.shape} but critic has shape {w.shape}")
    rng_state = int(seed) % 2**32

    def normal():
        nonlocal rng_state
        rng_state = (1664525 * rng_state + 1013904223) % 2**32
        u1 = (rng_state + 0.5) / 2**32
        rng_state = (1664525 * rng_state + 1013904223) % 2**32
        u2 = (rng_state + 0.5) / 2**32
        return float(np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2))

    noise = 0.0
    losses, rewards, actions, noises, qs = [], [], [], [], []
    for _ in range(E):
        mu = float(th @ s)
        noise = noise + ou_theta * (0.0 - noise) + ou_sigma * normal()
        a = mu + noise
        out = env(s, a)
        if not (isinstance(out, (tuple, list)) and len(out) == 3):
            raise ValueError("geron_ddpg: env must return exactly (s2, r, done)")
        s2, rew, done = out
        s2 = np.atleast_1d(np.asarray(s2, dtype=float))
        if s2.size != s.size:
            raise ValueError(f"geron_ddpg: env returned a state of size {s2.size}, expected {s.size}")
        rew = float(rew)
        if not np.isfinite(rew):
            raise ValueError("geron_ddpg: env returned a non-finite reward")

        feat = np.concatenate([s, [a]])
        q = float(w @ feat)
        a2 = float(th_t @ s2)
        q2 = 0.0 if bool(done) else float(w_t @ np.concatenate([s2, [a2]]))
        td = rew + g * q2 - q
        w = w + eta * td * feat

        # Deterministic policy gradient: dQ/da * dmu/dtheta.
        dq_da = float(w[-1])
        th = th + eta * dq_da * s

        th_t = t * th + (1 - t) * th_t
        w_t = t * w + (1 - t) * w_t

        losses.append(float(td**2))
        rewards.append(rew)
        actions.append(float(a))
        noises.append(float(noise))
        qs.append(q)
        s = s2 if not done else s

    return RichResult(
        title="DDPG training",
        summary_lines=[("Epochs", E), ("Mean reward", float(np.mean(rewards))), ("Final critic loss", losses[-1])],
        interpretation="A deterministic policy explores only through its noise process, which is why OU noise is correlated.",
        payload={
            "actor": th.tolist(),
            "critic": w.tolist(),
            "actor_target": th_t.tolist(),
            "critic_target": w_t.tolist(),
            "critic_losses": losses,
            "rewards": rewards,
            "actions": actions,
            "ou_noise": noises,
            "q_values": qs,
            "gamma": g,
            "tau": t,
            "estimate": float(np.mean(rewards)),
            "n": int(E),
            "method": "linear DDPG: TD critic, deterministic policy gradient actor, Polyak targets, OU exploration",
        },
    )


def cheatsheet():
    return "hmddpg: Deep deterministic policy gradient (DDPG)"


# compact alias per ledger/NAMING.md
geronddpg = geron_ddpg
