# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deep Q-Network loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_dqn_loss"]

_METHOD = "DQN bootstrap MSE loss"


def geron_dqn_loss(Q, Q_target, batch, gamma=0.99):
    r"""Squared error between the online Q and its bootstrap target.

    .. math::
        L = \mathbb E\Bigl[\bigl(r + \gamma \max_{a'}
        Q_{\text{target}}(s', a'; \theta^-)
        - Q(s, a; \theta)\bigr)^2\Bigr]

    The target uses the *frozen* network :math:`\theta^-`.  Bootstrapping
    off the network being trained makes the target move with every
    update, and the resulting feedback loop is what the target network
    exists to break -- so ``Q`` and ``Q_target`` are separate arguments
    here, and passing the same array twice is exactly the unstable
    configuration.

    Terminal transitions drop the bootstrap term entirely: after a
    terminal state there is no future return, and adding
    :math:`\gamma \max Q` there is the classic bug that inflates values
    near the end of an episode.

    Parameters
    ----------
    Q, Q_target : array-like, shape (n_states, n_actions)
        Online and frozen action-value tables.
    batch : sequence of tuples
        ``(s, a, r, s_next, done)``; ``done`` may be omitted and
        defaults to False.
    gamma : float, optional
        Discount in ``[0, 1]``, default 0.99.

    Returns
    -------
    RichResult
        Payload keys ``loss``, ``targets``, ``predictions``,
        ``td_errors``, ``max_abs_td_error``, ``n_terminal``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 19, DQN section (Mnih et al. 2015).

    Examples
    --------
    Zero tables, one transition with reward 1: the target is 1, the
    prediction 0, so the loss is 1.

    >>> Z = [[0.0, 0.0], [0.0, 0.0]]
    >>> r = geron_dqn_loss(Z, Z, [(0, 0, 1.0, 1, False)], gamma=0.9)
    >>> r["targets"], r["loss"]
    ([1.0], 1.0)

    Give the frozen network a value at the next state and the discount
    shows up -- ``1 + 0.9 * 10``:

    >>> QT = [[0.0, 0.0], [10.0, 3.0]]
    >>> geron_dqn_loss(Z, QT, [(0, 0, 1.0, 1, False)], gamma=0.9)["targets"]
    [10.0]

    Mark the transition terminal and the bootstrap disappears:

    >>> geron_dqn_loss(Z, QT, [(0, 0, 1.0, 1, True)], gamma=0.9)["targets"]
    [1.0]
    """
    Qa = np.atleast_2d(np.asarray(Q, dtype=float))
    Qt = np.atleast_2d(np.asarray(Q_target, dtype=float))
    if Qa.shape != Qt.shape:
        raise ValueError(f"Q {Qa.shape} and Q_target {Qt.shape} must have the same shape.")
    if Qa.ndim != 2 or Qa.size == 0:
        raise ValueError(f"Q must be a non-empty (n_states, n_actions) table, got {Qa.shape}.")
    if not np.all(np.isfinite(Qa)) or not np.all(np.isfinite(Qt)):
        raise ValueError("Q and Q_target must be finite.")
    nS, nA = Qa.shape
    gamma = float(gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}.")
    rows = list(batch)
    if not rows:
        raise ValueError("batch is empty.")

    targets, preds, n_term = [], [], 0
    for k, tr in enumerate(rows):
        if len(tr) == 4:
            s, a, r, s2 = tr
            done = False
        elif len(tr) == 5:
            s, a, r, s2, done = tr
        else:
            raise ValueError(
                f"batch[{k}] must be (s, a, r, s_next[, done]), got {len(tr)} items."
            )
        s, a, s2 = int(s), int(a), int(s2)
        for name, v, hi in (("s", s, nS), ("s_next", s2, nS), ("a", a, nA)):
            if not (0 <= v < hi):
                raise ValueError(f"batch[{k}]: {name} = {v} is outside [0, {hi - 1}].")
        r = float(r)
        if not np.isfinite(r):
            raise ValueError(f"batch[{k}]: reward must be finite, got {r}.")
        boot = 0.0 if done else gamma * float(Qt[s2].max())
        if done:
            n_term += 1
        targets.append(r + boot)
        preds.append(float(Qa[s, a]))

    tgt = np.asarray(targets)
    prd = np.asarray(preds)
    td = tgt - prd
    loss = float(np.mean(td**2))

    return RichResult(
        title="DQN loss",
        summary_lines=[("Loss", loss), ("Transitions", len(rows)),
                       ("Terminal", n_term)],
        payload={
            "loss": loss,
            "targets": tgt.tolist(),
            "predictions": prd.tolist(),
            "td_errors": td.tolist(),
            "max_abs_td_error": float(np.max(np.abs(td))),
            "n_terminal": n_term,
            "gamma": gamma,
            "estimate": loss,
            "n": len(rows),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grdqnl: L = mean((r + gamma max_a' Q_target(s') - Q(s,a))^2); terminal drops the bootstrap"
