# morie.fn -- function file (rootcoder007/morie)
"""LARS -- layer-wise adaptive rate scaling, You, Gitman & Ginsburg (2017)."""

from __future__ import annotations

from . import _array_core as np

from ._optim import as_vector, init_state, step_result

__all__ = ["lars_optimizer"]


def lars_optimizer(g, w, lr=0.1, mu=0.9, wd=0.0, eta=0.001, eps=1e-8, state=None):
    r"""One LARS step for a single layer's parameters.

    Large-batch training breaks when one layer's ratio of weight norm to
    gradient norm differs wildly from another's, because a single global
    learning rate cannot suit both. LARS rescales each layer's step by a
    local trust ratio

    .. math::
        \gamma = \eta\, \frac{\lVert w \rVert}{\lVert g \rVert + \lambda \lVert w \rVert + \epsilon},

    then applies momentum to :math:`\gamma (g + \lambda w)`. The step length
    therefore tracks :math:`\lVert w \rVert`, so it is scale-free in the
    weights.

    Where :math:`\lVert w \rVert = 0` the trust ratio is undefined; LARS
    convention is to fall back to the unscaled step, which this follows.

    Parameters
    ----------
    g : array-like
        Gradient for this layer.
    w : array-like
        Current weights for this layer; same size as ``g``.
    lr : float
        Global learning rate.
    mu : float
        Momentum coefficient in [0, 1).
    wd : float
        Weight decay :math:`\lambda`, folded into the gradient (LARS, unlike
        AdamW, does *not* decouple it).
    eta : float
        Trust coefficient.
    eps : float
        Denominator floor.
    state : dict, optional
        ``state`` from the previous call.

    Returns
    -------
    RichResult
        ``update``, ``state``, ``trust_ratio``, ``v``.

    References
    ----------
    You, Y., Gitman, I., & Ginsburg, B. (2017). Large batch training of
        convolutional networks. arXiv:1708.03888.

    Examples
    --------
    The trust ratio scales with the weight norm, so doubling the weights
    doubles the step.

    >>> import numpy as np
    >>> a = lars_optimizer([1.0, 1.0], [1.0, 1.0], lr=1.0, mu=0.0)["update"]
    >>> b = lars_optimizer([1.0, 1.0], [2.0, 2.0], lr=1.0, mu=0.0)["update"]
    >>> bool(abs(np.linalg.norm(b) - 2 * np.linalg.norm(a)) < 1e-9)
    True

    Zero weights fall back to the unscaled step rather than dividing by zero.

    >>> r = lars_optimizer([1.0], [0.0], lr=0.1, mu=0.0)
    >>> float(r["trust_ratio"]), float(round(r["update"][0], 6))
    (1.0, -0.1)
    """
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must be in [0, 1)")
    g = as_vector(g)
    w = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    if w.size != g.size:
        raise ValueError(f"w has {w.size} entries but g has {g.size}")
    wn = float(np.linalg.norm(w))
    gn = float(np.linalg.norm(g))
    trust = eta * wn / (gn + wd * wn + eps) if wn > 0 and gn > 0 else 1.0
    st = init_state(state, g.size, keys=("v",))
    st["v"] = mu * st["v"] + trust * (g + wd * w)
    update = -lr * st["v"]
    return step_result(
        update, st, "LARS", trust_ratio=float(trust), w_norm=wn, g_norm=gn, v=st["v"]
    )


def cheatsheet():
    return "larspec: LARS -- per-layer trust ratio eta*||w||/||g||; step scales with the weight norm"


# compact alias per ledger/NAMING.md
larsoptimizer = lars_optimizer
