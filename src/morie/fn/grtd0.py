# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TD(0) value update."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_td_zero_update"]

_METHOD = "TD(0) value update"


def geron_td_zero_update(V, state, next_state, reward, alpha, gamma, done=False):
    r"""One-step temporal-difference update of a state-value table.

    .. math::
        V(S_t) \leftarrow V(S_t) + \alpha\bigl[
            r + \gamma V(S_{t+1}) - V(S_t)\bigr]

    TD(0) bootstraps: it corrects a guess with a guess, one step ahead.
    That is why it can learn online from an unfinished episode, where
    Monte Carlo has to wait for the return.  The bracketed quantity is
    the TD error, and its sign is all the information the update carries
    -- a positive error means the state was worth more than the table
    said.

    Parameters
    ----------
    V : array-like, shape (n_states,)
    state, next_state : int
    reward : float
    alpha : float
        Learning rate in ``(0, 1]``.
    gamma : float
        Discount in ``[0, 1]``.
    done : bool, optional
        Terminal transition: target is ``reward`` alone.

    Returns
    -------
    RichResult
        Payload keys ``V``, ``old_value``, ``new_value``, ``target``,
        ``td_error``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Temporal Difference Learning section.

    Examples
    --------
    ``V = [0, 5]``, reward 1, ``gamma = 0.9``: target ``1 + 4.5 = 5.5``,
    error 5.5, and ``alpha = 0.2`` moves ``V(0)`` to 1.1.

    >>> r = geron_td_zero_update([0.0, 5.0], 0, 1, 1.0, alpha=0.2, gamma=0.9)
    >>> r["target"], r["td_error"]
    (5.5, 5.5)
    >>> round(r["new_value"], 10)
    1.1

    A zero TD error leaves the table untouched -- the fixed point:

    >>> geron_td_zero_update([5.5, 5.0], 0, 1, 1.0, alpha=0.2, gamma=0.9)["td_error"]
    0.0
    """
    Vv = np.asarray(V, dtype=float).ravel().copy()
    if Vv.size == 0:
        raise ValueError("V is empty.")
    if not np.all(np.isfinite(Vv)):
        raise ValueError("V contains non-finite values.")
    s, sn = int(state), int(next_state)
    if not (0 <= s < Vv.size) or not (0 <= sn < Vv.size):
        raise ValueError(f"states must lie in [0, {Vv.size - 1}], got {s} and {sn}.")
    reward = float(reward)
    if not np.isfinite(reward):
        raise ValueError(f"reward must be finite, got {reward}.")
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must lie in (0, 1], got {alpha}.")
    gamma = float(gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}.")

    old = float(Vv[s])
    target = reward if done else reward + gamma * float(Vv[sn])
    td = target - old
    Vv[s] = old + alpha * td

    return RichResult(
        title="TD(0) update",
        summary_lines=[("TD error", float(td)), ("V(s)", float(Vv[s]))],
        payload={
            "V": Vv.tolist(),
            "old_value": old,
            "new_value": float(Vv[s]),
            "target": float(target),
            "td_error": float(td),
            "estimate": float(Vv[s]),
            "n": int(Vv.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grtd0: V(s) += alpha[r + gamma V(s') - V(s)]; bootstraps, so it learns mid-episode"
