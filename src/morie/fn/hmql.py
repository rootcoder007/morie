# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Q-learning: off-policy temporal-difference control."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_q_learning"]


def geron_q_learning(Q, s, a, r, s_next, alpha, gamma, done=False):
    """
    Q-learning: off-policy TD control.

    Formula: Q(s,a) <- Q(s,a) + alpha*(r + gamma*max_{a'} Q(s',a') - Q(s,a))

    The target uses the MAXIMUM over next actions, not the action the
    behaviour policy will actually take -- that is what makes Q-learning
    off-policy, and what lets it learn the optimal action-values while
    exploring randomly. The same max is also the source of its
    optimistic bias: noise in Q(s', .) is maximised over, so every
    estimate is pulled upward, which is the defect Double DQN exists to
    fix.

    On a terminal transition (``done=True``) the bootstrap term is
    dropped: there is no next state to be worth anything.

    Parameters
    ----------
    Q : array-like, shape (n_states, n_actions)
        Action-value table, updated out of place.
    s, a : int
        Current state and action taken.
    r : float
        Reward received.
    s_next : int
        Resulting state (ignored when ``done``).
    alpha : float
        Learning rate in (0, 1].
    gamma : float
        Discount in [0, 1].
    done : bool, default False
        Whether ``s_next`` is terminal.

    Returns
    -------
    result : RichResult
        Keys: Q, td_error, target, old_value, new_value, estimate, n,
        method.

    Examples
    --------
    Target = 1 + 0.9 * max(0, 1) = 1.9, so Q(0,0) moves halfway to it:

    >>> r = geron_q_learning([[0.0, 0.0], [0.0, 1.0]], 0, 0, 1.0, 1, 0.5, 0.9)
    >>> float(r["target"]), float(r["td_error"])
    (1.9, 1.9)
    >>> float(r["new_value"])
    0.95

    A terminal transition drops the bootstrap:

    >>> float(geron_q_learning([[0.0, 0.0], [0.0, 1.0]], 0, 0, 1.0, 1, 1.0, 0.9,
    ...                        done=True)["new_value"])
    1.0

    References
    ----------
    Geron Ch 19
    """
    T = np.array(np.asarray(Q, dtype=float), copy=True)
    if T.ndim != 2:
        raise ValueError(f"geron_q_learning: Q must be a 2-D (states, actions) table, got ndim={T.ndim}")
    ns, na = T.shape
    si, ai = int(s), int(a)
    if not (0 <= si < ns):
        raise ValueError(f"geron_q_learning: state {si} outside the {ns} rows of Q")
    if not (0 <= ai < na):
        raise ValueError(f"geron_q_learning: action {ai} outside the {na} columns of Q")
    al, ga = float(alpha), float(gamma)
    if not (0.0 < al <= 1.0):
        raise ValueError(f"geron_q_learning: alpha must lie in (0, 1], got {alpha!r}")
    if not (0.0 <= ga <= 1.0):
        raise ValueError(f"geron_q_learning: gamma must lie in [0, 1], got {gamma!r}")
    rr = float(r)
    if not np.isfinite(rr):
        raise ValueError("geron_q_learning: r must be finite")

    if done:
        target = rr
        best_next = float("nan")
    else:
        sn = int(s_next)
        if not (0 <= sn < ns):
            raise ValueError(f"geron_q_learning: next state {sn} outside the {ns} rows of Q")
        best_next = float(np.max(T[sn]))
        target = rr + ga * best_next
    old = float(T[si, ai])
    td = target - old
    T[si, ai] = old + al * td
    return RichResult(
        title="Q-learning update",
        summary_lines=[("TD error", td), ("Q(s,a)", float(T[si, ai]))],
        interpretation="The max over next actions makes this off-policy and biases the estimate upward.",
        payload={
            "Q": T,
            "td_error": float(td),
            "target": float(target),
            "old_value": old,
            "new_value": float(T[si, ai]),
            "max_next": best_next,
            "estimate": float(T[si, ai]),
            "n": int(T.size),
            "method": "Tabular Q-learning update (off-policy TD control)",
        },
    )


def cheatsheet():
    return "hmql: Q-learning off-policy TD control update"
