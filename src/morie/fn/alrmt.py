# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bradley-Terry reward-model loss (Alammar Ch 12; RLHF)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_reward_model_training_bt"]


def alammar_reward_model_training_bt(scores_w, scores_l):
    """L = -mean log sigmoid(r(y_w) - r(y_l)).

    A pair the model orders wrongly contributes more than log 2; the
    payload reports the accuracy alongside, since a loss below log 2
    on average is exactly the claim that most pairs are ordered right.

    References: Alammar and Grootendorst, Ch 12; Bradley and Terry
    (1952); Ouyang et al. (2022).

    Examples
    --------
    >>> out = alammar_reward_model_training_bt([2.0], [0.0])
    >>> round(out["estimate"], 6)
    0.126928
    """
    rw = np.atleast_1d(np.asarray(scores_w, dtype=float))
    rl = np.atleast_1d(np.asarray(scores_l, dtype=float))
    if rw.shape != rl.shape:
        raise ValueError("need one loser score per winner score.")
    diff = rw - rl
    losses = np.logaddexp(0.0, -diff)      # -log sigmoid(diff), stable
    return RichResult(payload={
        "estimate": float(losses.mean()),
        "losses": [float(v) for v in losses],
        "pair_accuracy": float(np.mean(diff > 0)), "n": len(diff),
        "method": "Bradley-Terry reward loss (Ouyang et al. 2022)"})


def cheatsheet():
    return "alrmt: -log sigmoid(r_w - r_l), pair accuracy reported"
