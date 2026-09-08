# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Double DQN: decouple action selection and evaluation."""

from . import _array_core as np

from ._richresult import RichResult
from .grddqn import geron_double_dqn_target
from .hmdqn import check_buffer

__all__ = ["geron_double_dqn"]


def geron_double_dqn(env, Q, Q_target, buffer, epochs=10, lr=0.1, gamma=0.95, target_sync=5, batch_size=None):
    """
    Double DQN: decouple action selection and evaluation.

    Formula: a* = argmax_a Q(s',a); target = r + gamma*Q_target(s', a*)

    The target computation is DELEGATED to
    :func:`morie.fn.grddqn.geron_double_dqn_target`, which implements the
    argmax/evaluate split exactly. This module wraps it in the training
    loop -- replay mini-batches and a periodically synced target net --
    and, on every epoch, records the vanilla DQN target alongside so the
    overestimation being removed is measurable rather than folklore.

    Vanilla DQN uses ``max_a Q_target(s', a)``, which takes both the
    choice and its valuation from the same noisy estimate and is therefore
    biased upward. Double DQN chooses with the online net and values with
    the target net; ``overestimation_gap`` per epoch is the difference,
    and it is non-negative whenever the two nets disagree.

    Parameters
    ----------
    env : callable or None
        Kept for provenance; learning uses the buffer.
    Q, Q_target : array-like, shape (S, A)
    buffer : sequence of (s, a, r, s2[, done])
    epochs : int, default 10
    lr : float, default 0.1
    gamma : float, default 0.95
    target_sync : int, default 5
    batch_size : int, optional

    Returns
    -------
    result : RichResult
        Keys: Q, Q_target, loss_history, targets, vanilla_targets,
        overestimation_gap, greedy_policy, sync_epochs, estimate, n,
        method.

    Examples
    --------
    The online net prefers action 1 while the target net thinks action 1
    is terrible. Double DQN believes the evaluation; vanilla DQN takes the
    optimistic max, and the gap is 20:

    >>> r = geron_double_dqn(None, [[0.0, 1.0]], [[10.0, -10.0]],
    ...                      [(0, 0, 0.0, 0, False)], epochs=1, lr=1.0, gamma=1.0)
    >>> r["targets"][0]
    -10.0
    >>> r["vanilla_targets"][0]
    10.0
    >>> r["overestimation_gap"][0]
    20.0
    >>> round(r["Q"][0][0], 6)
    -10.0

    With a terminal successor the bootstrap disappears and both agree:

    >>> r2 = geron_double_dqn(None, [[0.0, 1.0]], [[10.0, -10.0]],
    ...                       [(0, 0, 5.0, 0, True)], epochs=1, lr=1.0, gamma=1.0)
    >>> r2["targets"][0], r2["overestimation_gap"][0]
    (5.0, 0.0)

    References
    ----------
    Géron Ch 19
    """
    Qa = np.array(Q, dtype=float)
    Qt = np.array(Q_target, dtype=float)
    if Qa.ndim != 2 or Qa.size == 0:
        raise ValueError(f"geron_double_dqn: Q must be a non-empty (S, A) table, got shape {Qa.shape}")
    if Qt.shape != Qa.shape:
        raise ValueError(f"geron_double_dqn: Q_target has shape {Qt.shape} but Q has shape {Qa.shape}")
    S, A = Qa.shape
    s, a, r, s2, done = check_buffer(buffer, S, A, "geron_double_dqn")
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_double_dqn: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"geron_double_dqn: lr must lie in (0, 1], got {lr!r}")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_double_dqn: gamma must lie in [0, 1], got {gamma!r}")
    sync = int(target_sync)
    if sync < 1:
        raise ValueError(f"geron_double_dqn: target_sync must be >= 1, got {target_sync!r}")
    N = s.size
    bs = N if batch_size is None else int(batch_size)
    if not (1 <= bs <= N):
        raise ValueError(f"geron_double_dqn: batch_size must lie in 1..{N}, got {batch_size!r}")

    hist, syncs = [], []
    targets = vanilla = gaps = None
    pos = 0
    for ep in range(E):
        idx = np.arange(pos, pos + bs) % N
        pos = (pos + bs) % N
        step = geron_double_dqn_target(Qa, Qt, s_next=s2[idx], r=r[idx], gamma=g, done=done[idx])
        tgt = np.asarray(step["target"], dtype=float)
        td = tgt - Qa[s[idx], a[idx]]
        for k, j in enumerate(idx):
            Qa[s[j], a[j]] += eta * td[k]
        hist.append(float(np.mean(td**2)))
        targets = tgt.tolist()
        vanilla = list(step["vanilla_target"])
        gaps = list(step["overestimation_gap"])
        if (ep + 1) % sync == 0:
            Qt = Qa.copy()
            syncs.append(ep + 1)

    return RichResult(
        title="Double DQN training",
        summary_lines=[("Epochs", E), ("Final TD loss", hist[-1]), ("Mean gap", float(np.mean(gaps)))],
        interpretation="Selecting with the online net and valuing with the target net removes the max-operator bias.",
        payload={
            "Q": Qa.tolist(),
            "Q_target": Qt.tolist(),
            "loss_history": hist,
            "targets": targets,
            "vanilla_targets": vanilla,
            "overestimation_gap": gaps,
            "greedy_policy": Qa.argmax(axis=1).astype(int).tolist(),
            "sync_epochs": syncs,
            "gamma": g,
            "lr": eta,
            "env": repr(env) if env is not None else None,
            "estimate": float(hist[-1]),
            "n": int(N),
            "method": "double DQN training loop; target computation delegated to grddqn",
        },
    )


def cheatsheet():
    return "hmddqn: Double DQN: decouple action selection and evaluation"


# compact alias per ledger/NAMING.md
gerondoubledqn = geron_double_dqn
