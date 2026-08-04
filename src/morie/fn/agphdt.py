# morie.fn -- slice s03 (rootcoder007/morie)
"""AlphaZero policy head.

Source consulted: Silver, D. et al. (2017), *Nature* 550, 354-359,
methods section "Neural network architecture": a 1x1 convolution to two
feature planes, batch normalisation, a rectifier, then a fully connected
layer to the action space plus one (the pass move in Go), whose outputs
are the *logits* of the move distribution.  Silver et al. (2018),
arXiv:1712.01815 (FETCHED), keeps the same head.  The Nature paper is
paywalled; the layer list is reproduced identically everywhere, and the
only numeric content -- that the head emits logits which are softmaxed
over legal moves -- is unambiguous.

Implemented as what it computes: logits (a linear map of the flattened
planes, or the planes themselves when no weights are supplied), an
illegal-move mask, and a softmax over what survives.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_policy_head"]


def alphazero_policy_head(x, action_space=None, W=None, legal=None):
    """Logits to a masked move distribution.

    Parameters
    ----------
    x : array-like
        The feature planes, flattened.
    action_space : int, optional
        Number of actions; defaults to the number of logits produced.
    W : 2-D array-like, optional
        Weights of the fully connected layer, one row per action.  With
        none supplied the first ``action_space`` flattened units are
        taken as the logits directly.
    legal : array-like, optional
        Legal-move mask.

    Returns
    -------
    RichResult with payload:
        estimate : probability of action 0
        p        : the move distribution
        logits   : the pre-softmax logits
        entropy  : Shannon entropy of p, in nats
    """
    f = k.vec(x)
    if W is not None:
        rows = k.mat(W)
        logits = k.matvec(rows, f)
    else:
        m = int(action_space) if action_space is not None else len(f)
        logits = f[:m] + [0.0] * (m - len(f)) if len(f) < m else f[:m]
    m = len(logits)
    if legal is not None:
        mask = [1.0 if q else 0.0 for q in legal]
        shifted = [logits[i] if mask[i] > 0.0 else float("-inf") for i in range(m)]
    else:
        shifted = list(logits)
    p = k.softmax([q for q in shifted if q != float("-inf")])
    out = []
    j = 0
    for i in range(m):
        if shifted[i] == float("-inf"):
            out.append(0.0)
        else:
            out.append(p[j])
            j += 1
    h = 0.0
    for q in out:
        if q > 0.0:
            h -= q * math.log(q)
    return RichResult(
        title="AlphaZero policy head",
        summary_lines=[("actions", m)],
        payload={
            "estimate": out[0] if out else float("nan"),
            "p": out,
            "logits": logits,
            "entropy": h,
            "method": "AlphaZero policy head: linear logits, mask, softmax",
        },
    )


def cheatsheet():
    return "agphdt: AlphaZero policy head"
