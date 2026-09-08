# morie.fn -- slice s03 (rootcoder007/morie)
"""One SGD step with momentum and L2 weight decay, as used to train AlphaZero.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815:
the parameters are "updated by stochastic gradient descent with
momentum", and the loss of Silver et al. (2017), *Nature* 550, 354-359,
carries an explicit L2 term c ||theta||^2 -- fetched and reproduced in
the AlphaZero paper as

    l = (z - v)^2 - pi^T log p + c ||theta||^2.

Because that penalty sits inside the loss, its gradient is 2 c theta and
so the update is the classical

    v_(t+1)     = mu v_t + (g_t + lambda theta_t)
    theta_(t+1) = theta_t - lr v_(t+1)

with lambda = 2c.  This is the "L2 regularisation" form, in which decay
passes through the momentum buffer, *not* the decoupled AdamW-style form
of Loshchilov and Hutter -- the two differ whenever mu is nonzero, and
the loss-based penalty of the paper is unambiguously the former.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_optimizer"]


def alphazero_optimizer(theta, grad, momentum=0.9, weight_decay=1e-4,
                        lr=0.2, buf=None):
    """A single SGD-with-momentum step under an L2 penalty.

    Parameters
    ----------
    theta : array-like
        Current parameters, flattened.
    grad : array-like
        Gradient of the data term at ``theta``.
    momentum : float
        Momentum coefficient mu.
    weight_decay : float
        lambda = 2c, the gradient of the L2 penalty per unit theta.
    lr : float
        Learning rate.
    buf : array-like, optional
        Previous momentum buffer; zeros by default.

    Returns
    -------
    RichResult with payload:
        estimate  : the first updated parameter
        theta_new : updated parameters
        buf       : the new momentum buffer
        step_norm : L2 norm of the applied update
    """
    th = k.vec(theta)
    g = k.vec(grad)
    b = k.vec(buf) if buf is not None else [0.0] * len(th)
    mu = float(momentum)
    wd = float(weight_decay)
    a = float(lr)
    nb = [0.0] * len(th)
    nt = [0.0] * len(th)
    s2 = 0.0
    for i in range(len(th)):
        nb[i] = mu * b[i] + (g[i] + wd * th[i])
        step = a * nb[i]
        nt[i] = th[i] - step
        s2 += step * step
    return RichResult(
        title="AlphaZero optimizer step",
        summary_lines=[("lr", a), ("momentum", mu)],
        payload={
            "estimate": nt[0] if nt else float("nan"),
            "theta_new": nt,
            "buf": nb,
            "step_norm": s2 ** 0.5,
            "method": "SGD with momentum and L2 weight decay (AlphaZero training)",
        },
    )


def cheatsheet():
    return "agclop: AlphaZero L2-regularized optimizer step"
