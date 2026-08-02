# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deep Q-network (DQN): neural Q-function with replay buffer and target net."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_dqn", "check_buffer"]


def check_buffer(buffer, n_states, n_actions, name):
    """Validate a replay buffer of ``(s, a, r, s2, done)`` and return arrays."""
    if buffer is None or len(buffer) == 0:
        raise ValueError(f"{name}: buffer is empty; DQN cannot learn without transitions")
    s, a, r, s2, d = [], [], [], [], []
    for i, tr in enumerate(buffer):
        if len(tr) not in (4, 5):
            raise ValueError(f"{name}: transition {i} has {len(tr)} fields, expected (s, a, r, s2[, done])")
        s.append(int(tr[0]))
        a.append(int(tr[1]))
        r.append(float(tr[2]))
        s2.append(int(tr[3]))
        d.append(bool(tr[4]) if len(tr) == 5 else False)
    s, a, s2 = np.asarray(s), np.asarray(a), np.asarray(s2)
    if s.min() < 0 or s.max() >= n_states or s2.min() < 0 or s2.max() >= n_states:
        raise ValueError(f"{name}: a state index is outside 0..{n_states - 1}")
    if a.min() < 0 or a.max() >= n_actions:
        raise ValueError(f"{name}: an action index is outside 0..{n_actions - 1}")
    return s, a, np.asarray(r), s2, np.asarray(d)


def geron_dqn(env, Q, Q_target, buffer, epochs=10, lr=0.1, gamma=0.95, target_sync=5, batch_size=None):
    """
    Deep Q-network (DQN): neural Q-function with replay buffer and target net.

    Formula: L = (r + gamma*max_a Q_target(s',a) - Q(s,a))^2

    A real training loop over a tabular Q-function -- tabular because that
    keeps the two ideas DQN actually contributes visible and checkable:

    * the **replay buffer** decorrelates updates, so transitions are
      replayed in mini-batches instead of in the order they occurred;
    * the **target network** freezes the bootstrap value for
      ``target_sync`` epochs, which is what stops the regression target
      chasing its own prediction.

    Each update is a gradient step on the squared TD error, which for a
    table is ``Q[s,a] += lr * td_error``. ``loss_history`` is the mean
    squared TD error per epoch and ``sync_epochs`` records when the target
    was refreshed -- the loss usually jumps at exactly those points, which
    is the target net doing its job rather than a bug.

    ``env`` is optional and unused for learning; when given as a callable
    it is only recorded, since the buffer already holds the experience.

    Parameters
    ----------
    env : callable or None
        Environment, kept for provenance only.
    Q, Q_target : array-like, shape (S, A)
        Online and target Q-tables (copied, not mutated).
    buffer : sequence of (s, a, r, s2[, done])
    epochs : int, default 10
    lr : float, default 0.1
        Step size in (0, 1].
    gamma : float, default 0.95
    target_sync : int, default 5
        Epochs between copying Q into Q_target.
    batch_size : int, optional
        Transitions sampled per update; default the whole buffer.

    Returns
    -------
    result : RichResult
        Keys: Q, Q_target, loss_history, td_errors, greedy_policy,
        sync_epochs, n_updates, estimate, n, method.

    Examples
    --------
    A single deterministic transition with a terminal successor: the
    target is just the reward, so one step of size 0.5 moves Q halfway.

    >>> Q = [[0.0, 0.0]]
    >>> r = geron_dqn(None, Q, Q, [(0, 0, 1.0, 0, True)], epochs=1, lr=0.5)
    >>> round(r["Q"][0][0], 6)
    0.5
    >>> round(r["loss_history"][0], 6)
    1.0

    Repeating the update converges to the reward, and the untouched action
    stays put:

    >>> r2 = geron_dqn(None, Q, Q, [(0, 0, 1.0, 0, True)], epochs=20, lr=0.5)
    >>> round(r2["Q"][0][0], 6)
    0.999999
    >>> r2["Q"][0][1]
    0.0
    >>> r2["loss_history"][-1] < r2["loss_history"][0]
    True

    Bootstrapping through a non-terminal successor uses the frozen target
    net, not the online one:

    >>> r3 = geron_dqn(None, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 10.0]],
    ...                [(0, 0, 0.0, 1, False)], epochs=1, lr=1.0, gamma=0.9)
    >>> round(r3["Q"][0][0], 6)
    9.0

    References
    ----------
    Géron Ch 19
    """
    Qa = np.array(Q, dtype=float)
    Qt = np.array(Q_target, dtype=float)
    if Qa.ndim != 2 or Qa.size == 0:
        raise ValueError(f"geron_dqn: Q must be a non-empty (S, A) table, got shape {Qa.shape}")
    if Qt.shape != Qa.shape:
        raise ValueError(f"geron_dqn: Q_target has shape {Qt.shape} but Q has shape {Qa.shape}")
    S, A = Qa.shape
    s, a, r, s2, done = check_buffer(buffer, S, A, "geron_dqn")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_dqn: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"geron_dqn: lr must lie in (0, 1], got {lr!r}")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_dqn: gamma must lie in [0, 1], got {gamma!r}")
    sync = int(target_sync)
    if sync < 1:
        raise ValueError(f"geron_dqn: target_sync must be >= 1, got {target_sync!r}")
    N = s.size
    bs = N if batch_size is None else int(batch_size)
    if not (1 <= bs <= N):
        raise ValueError(f"geron_dqn: batch_size must lie in 1..{N}, got {batch_size!r}")

    hist, syncs, td_last = [], [], None
    updates = 0
    pos = 0
    for ep in range(E):
        idx = np.arange(pos, pos + bs) % N
        pos = (pos + bs) % N
        boot = np.where(done[idx], 0.0, g * Qt[s2[idx]].max(axis=1))
        target = r[idx] + boot
        td = target - Qa[s[idx], a[idx]]
        for k, j in enumerate(idx):
            Qa[s[j], a[j]] += eta * td[k]
            updates += 1
        hist.append(float(np.mean(td**2)))
        td_last = td.tolist()
        if (ep + 1) % sync == 0:
            Qt = Qa.copy()
            syncs.append(ep + 1)

    return RichResult(
        title="DQN training",
        summary_lines=[("Epochs", E), ("Updates", updates), ("Final TD loss", hist[-1])],
        interpretation="The target net freezes the bootstrap, so the regression target stops chasing the prediction.",
        payload={
            "Q": Qa.tolist(),
            "Q_target": Qt.tolist(),
            "loss_history": hist,
            "td_errors": td_last,
            "greedy_policy": Qa.argmax(axis=1).astype(int).tolist(),
            "state_values": Qa.max(axis=1).tolist(),
            "sync_epochs": syncs,
            "n_updates": int(updates),
            "gamma": g,
            "lr": eta,
            "env": repr(env) if env is not None else None,
            "estimate": float(hist[-1]),
            "n": int(N),
            "method": "tabular DQN with replay mini-batches and a periodically synced target network",
        },
    )


def cheatsheet():
    return "hmdqn: Deep Q-network (DQN): neural Q-function with replay buffer and target net"
