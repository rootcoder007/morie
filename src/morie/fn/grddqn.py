# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Double DQN decouples action selection from evaluation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_double_dqn_target"]

_METHOD = "Double DQN target"


def geron_double_dqn_target(Q_online, Q_target, s_next, r, gamma, done=None):
    r"""Compute Double-DQN TD targets.

    .. math::
        y = r + \gamma\, Q_{\theta^-}\bigl(s',\,
        \arg\max_{a'} Q_{\theta}(s', a')\bigr)

    Vanilla DQN uses :math:`\max_{a'} Q_{\theta^-}(s',a')`, which takes a
    max over noisy estimates and is therefore biased *upwards* -- the
    max of noise is not the noise of the max.  Double DQN lets the online
    net choose the action and the target net price it, so an action that
    is overrated by one network is only used if the other agrees.  The
    ``overestimation_gap`` returned here is exactly the bias removed.

    Parameters
    ----------
    Q_online, Q_target : array-like, shape (S, A)
        Online and target Q-tables.
    s_next : array-like of int
        Successor state indices.
    r : array-like
        Rewards, one per transition.
    gamma : float
        Discount factor in ``[0, 1]``.
    done : array-like of bool, optional
        Terminal flags; the bootstrap is dropped where true.

    Returns
    -------
    RichResult
        Payload keys ``target``, ``selected_action``,
        ``vanilla_target``, ``overestimation_gap``, ``estimate`` (mean
        target), ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Double DQN section.

    Examples
    --------
    The online net prefers action 1; the target net thinks action 1 is
    terrible.  Double DQN believes the evaluation, vanilla DQN takes the
    optimistic max:

    >>> r = geron_double_dqn_target([[0.0, 1.0]], [[10.0, -10.0]],
    ...                             s_next=[0], r=[0.0], gamma=1.0)
    >>> r["selected_action"]
    [1]
    >>> r["target"]
    [-10.0]
    >>> r["vanilla_target"]
    [10.0]
    >>> r["overestimation_gap"]
    [20.0]

    Terminal transitions drop the bootstrap entirely:

    >>> geron_double_dqn_target([[0.0, 1.0]], [[10.0, -10.0]], [0], [5.0],
    ...                         1.0, done=[True])["target"]
    [5.0]
    """
    Qo = np.atleast_2d(np.asarray(Q_online, dtype=float))
    Qt = np.atleast_2d(np.asarray(Q_target, dtype=float))
    if Qo.shape != Qt.shape:
        raise ValueError(
            f"Q_online shape {Qo.shape} must match Q_target shape {Qt.shape}."
        )
    if Qo.size == 0:
        raise ValueError("Q tables are empty.")
    if not np.all(np.isfinite(Qo)) or not np.all(np.isfinite(Qt)):
        raise ValueError("Q tables must be finite.")
    S, A = Qo.shape
    sn = np.asarray(s_next).ravel()
    rew = np.asarray(r, dtype=float).ravel()
    if sn.size != rew.size:
        raise ValueError(
            f"s_next and r must have equal length, got {sn.size} and {rew.size}."
        )
    if sn.size == 0:
        raise ValueError("no transitions supplied.")
    sn = sn.astype(int)
    if sn.min() < 0 or sn.max() >= S:
        raise ValueError(f"s_next indices must lie in [0, {S - 1}].")
    if not np.all(np.isfinite(rew)):
        raise ValueError("r contains non-finite values.")
    gamma = float(gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}.")
    if done is None:
        cont = np.ones(sn.size)
    else:
        d = np.asarray(done).ravel()
        if d.size != sn.size:
            raise ValueError(
                f"done must have one flag per transition ({sn.size}), got {d.size}."
            )
        cont = 1.0 - d.astype(bool).astype(float)

    a_star = Qo[sn].argmax(axis=1)
    q_eval = Qt[sn, a_star]
    target = rew + gamma * cont * q_eval
    vanilla = rew + gamma * cont * Qt[sn].max(axis=1)

    return RichResult(
        title="Double DQN target",
        summary_lines=[("Mean target", float(target.mean())),
                       ("Mean overestimation removed", float((vanilla - target).mean()))],
        payload={
            "target": target.tolist(),
            "selected_action": a_star.tolist(),
            "q_eval": q_eval.tolist(),
            "vanilla_target": vanilla.tolist(),
            "overestimation_gap": (vanilla - target).tolist(),
            "gamma": gamma,
            "estimate": float(target.mean()),
            "n": int(sn.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grddqn: Double DQN y = r + gamma*Q_target(s', argmax_a Q_online(s', a))"
