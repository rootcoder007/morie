# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dueling DQN: separate value and advantage streams."""

from . import _array_core as np

from ._richresult import RichResult
from .hmdqn import check_buffer

__all__ = ["geron_dueling_dqn", "dueling_q"]


def dueling_q(V, Adv):
    """Combine the two streams: ``Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)``."""
    V = np.asarray(V, dtype=float).ravel()
    Adv = np.atleast_2d(np.asarray(Adv, dtype=float))
    if V.size != Adv.shape[0]:
        raise ValueError(f"dueling_q: V has {V.size} states but A has {Adv.shape[0]} rows")
    return V[:, None] + Adv - Adv.mean(axis=1, keepdims=True)


def geron_dueling_dqn(env, V, A, buffer, epochs=10, lr=0.1, gamma=0.95, target_sync=5):
    """
    Dueling DQN: separate value and advantage streams.

    Formula: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)

    Both streams are trained here, not just combined. The TD error on
    ``Q`` is backpropagated to the two heads through the aggregation
    above, whose derivatives are exact and worth stating: ``dQ/dV = 1``
    and ``dQ(s,a)/dA(s,a') = [a = a'] - 1/|A|``. So a step updates ``V``
    by the TD error and shifts the taken action's advantage relative to
    the others -- which is the whole point: the value of a state can be
    learned from any action, while advantages only need to encode
    differences.

    The mean-subtraction is what makes the decomposition identifiable. Two
    diagnostics are reported for it: ``advantage_mean`` (which stays 0 by
    construction) and ``value_share``, the fraction of ``|Q|`` explained
    by ``V`` alone.

    Parameters
    ----------
    env : callable or None
        Kept for provenance; learning uses the buffer.
    V : array-like, shape (S,)
        State-value stream.
    A : array-like, shape (S, nA)
        Advantage stream.
    buffer : sequence of (s, a, r, s2[, done])
    epochs : int, default 10
    lr : float, default 0.1
    gamma : float, default 0.95
    target_sync : int, default 5

    Returns
    -------
    result : RichResult
        Keys: Q, V, A, loss_history, advantage_mean, value_share,
        greedy_policy, sync_epochs, estimate, n, method.

    Examples
    --------
    The aggregation is mean-centred, so equal advantages leave Q equal to
    V and the advantage row always averages to zero:

    >>> r = geron_dueling_dqn(None, [2.0], [[1.0, 1.0]], [(0, 0, 2.0, 0, True)],
    ...                       epochs=1, lr=0.0)
    >>> r["Q"][0]
    [2.0, 2.0]
    >>> round(r["advantage_mean"], 12)
    0.0

    One terminal transition with TD error 1 and a full step: V rises by 1
    and the advantage head moves by ``[1 - 1/2, -1/2] = [0.5, -0.5]``, so
    Q(s,0) ends at 1.5 and Q(s,1) at 0.5.

    >>> r2 = geron_dueling_dqn(None, [0.0], [[0.0, 0.0]], [(0, 0, 1.0, 0, True)],
    ...                        epochs=1, lr=1.0)
    >>> round(r2["V"][0], 6)
    1.0
    >>> [round(v, 6) for v in r2["A"][0]]
    [0.5, -0.5]
    >>> [round(v, 6) for v in r2["Q"][0]]
    [1.5, 0.5]

    Training reduces the TD loss:

    >>> r3 = geron_dueling_dqn(None, [0.0], [[0.0, 0.0]], [(0, 0, 1.0, 0, True)],
    ...                        epochs=15, lr=0.2)
    >>> r3["loss_history"][-1] < r3["loss_history"][0]
    True

    References
    ----------
    Géron Ch 19
    """
    Vv = np.asarray(V, dtype=float).ravel().copy()
    Av = np.atleast_2d(np.asarray(A, dtype=float)).copy()
    if Vv.size == 0 or Av.size == 0:
        raise ValueError("geron_dueling_dqn: V and A must be non-empty")
    if Vv.size != Av.shape[0]:
        raise ValueError(f"geron_dueling_dqn: V has {Vv.size} states but A has {Av.shape[0]} rows")
    if not np.all(np.isfinite(Vv)) or not np.all(np.isfinite(Av)):
        raise ValueError("geron_dueling_dqn: V and A must be finite")
    S, nA = Av.shape
    s, a, r, s2, done = check_buffer(buffer, S, nA, "geron_dueling_dqn")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_dueling_dqn: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"geron_dueling_dqn: lr must lie in [0, 1], got {lr!r}")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_dueling_dqn: gamma must lie in [0, 1], got {gamma!r}")
    sync = int(target_sync)
    if sync < 1:
        raise ValueError(f"geron_dueling_dqn: target_sync must be >= 1, got {target_sync!r}")

    Q = dueling_q(Vv, Av)
    Qt = Q.copy()
    hist, syncs = [], []
    for ep in range(E):
        Q = dueling_q(Vv, Av)
        boot = np.where(done, 0.0, g * Qt[s2].max(axis=1))
        td = r + boot - Q[s, a]
        hist.append(float(np.mean(td**2)))
        for k in range(s.size):
            i, j, e = s[k], a[k], td[k]
            Vv[i] += eta * e
            grad = -np.full(nA, 1.0 / nA)
            grad[j] += 1.0
            Av[i] += eta * e * grad
        if (ep + 1) % sync == 0:
            Qt = dueling_q(Vv, Av)
            syncs.append(ep + 1)

    Q = dueling_q(Vv, Av)
    denom = float(np.mean(np.abs(Q)))
    share = float(np.mean(np.abs(np.repeat(Vv[:, None], nA, axis=1))) / denom) if denom > 0 else 1.0

    return RichResult(
        title="Dueling DQN",
        summary_lines=[("Epochs", E), ("Final TD loss", hist[-1]), ("Value share", share)],
        interpretation="Mean-centring the advantages makes the V/A split identifiable; V learns from every action.",
        payload={
            "Q": Q.tolist(),
            "V": Vv.tolist(),
            "A": Av.tolist(),
            "loss_history": hist,
            "advantage_mean": float(np.mean(Av - Av.mean(axis=1, keepdims=True))),
            "value_share": share,
            "greedy_policy": Q.argmax(axis=1).astype(int).tolist(),
            "sync_epochs": syncs,
            "gamma": g,
            "lr": eta,
            "env": repr(env) if env is not None else None,
            "estimate": float(hist[-1]),
            "n": int(s.size),
            "method": "dueling DQN with exact gradients through Q = V + A - mean(A)",
        },
    )


def cheatsheet():
    return "hmdldqn: Dueling DQN: separate value and advantage streams"
