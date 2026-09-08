# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Q-learning off-policy update."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_q_learning_update"]

_METHOD = "Q-learning update (off-policy TD control)"


def geron_q_learning_update(Q, s, a, r, s_next, alpha, gamma, done=False):
    r"""One off-policy temporal-difference update of a Q-table.

    .. math::
        Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[
            r + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr]

    The ``max`` is what makes this *off*-policy: the target assumes
    greedy behaviour from the next state, whatever the agent actually did
    there, so an epsilon-greedy explorer still converges to the optimal
    Q.  It is also where the overestimation bias comes from -- taking a
    max over noisy estimates is biased upward, which Double DQN
    (:mod:`morie.fn.grddqn`) exists to fix.  A terminal transition drops
    the bootstrap entirely; ``done=True`` is not cosmetic.

    Parameters
    ----------
    Q : array-like, shape (n_states, n_actions)
    s, a : int
    r : float
    s_next : int
    alpha : float
        Learning rate in ``(0, 1]``.
    gamma : float
        Discount in ``[0, 1]``.
    done : bool, optional
        Terminal transition: the target is just ``r``.

    Returns
    -------
    RichResult
        Payload keys ``Q`` (updated table), ``old_value``,
        ``new_value``, ``target``, ``td_error``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 19, Q-learning section.

    Examples
    --------
    ``Q(0,0) = 0``, reward 1, best next value 10, ``gamma = 0.9``:
    target ``1 + 9 = 10``, TD error 10, and with ``alpha = 0.1`` the
    entry moves to 1.

    >>> Qt = [[0.0, 0.0], [10.0, 5.0]]
    >>> r = geron_q_learning_update(Qt, 0, 0, 1.0, 1, alpha=0.1, gamma=0.9)
    >>> r["target"], r["td_error"]
    (10.0, 10.0)
    >>> round(r["new_value"], 10)
    1.0

    Terminal transitions do not bootstrap:

    >>> geron_q_learning_update(Qt, 0, 0, 1.0, 1, alpha=0.1, gamma=0.9, done=True)["target"]
    1.0
    """
    Qm = np.atleast_2d(np.asarray(Q, dtype=float)).copy()
    if Qm.ndim != 2 or Qm.size == 0:
        raise ValueError(f"Q must be a non-empty (n_states, n_actions) table, got shape {Qm.shape}.")
    if not np.all(np.isfinite(Qm)):
        raise ValueError("Q contains non-finite values.")
    S, A = Qm.shape
    s, a, s_next = int(s), int(a), int(s_next)
    if not (0 <= s < S) or not (0 <= s_next < S):
        raise ValueError(f"states must lie in [0, {S - 1}], got s={s}, s_next={s_next}.")
    if not (0 <= a < A):
        raise ValueError(f"action must lie in [0, {A - 1}], got {a}.")
    r = float(r)
    if not np.isfinite(r):
        raise ValueError(f"reward must be finite, got {r}.")
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}.")
    gamma = float(gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}.")

    old = float(Qm[s, a])
    best_next = float(Qm[s_next].max())
    target = r if done else r + gamma * best_next
    td = target - old
    Qm[s, a] = old + alpha * td

    return RichResult(
        title="Q-learning update",
        summary_lines=[("TD error", float(td)), ("Q(s,a)", float(Qm[s, a]))],
        payload={
            "Q": Qm.tolist(),
            "old_value": old,
            "new_value": float(Qm[s, a]),
            "target": float(target),
            "td_error": float(td),
            "max_next": best_next,
            "estimate": float(Qm[s, a]),
            "n": int(S),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grql: Q(s,a) += alpha[r + gamma max_a' Q(s',a') - Q(s,a)]; done=True drops the bootstrap"
