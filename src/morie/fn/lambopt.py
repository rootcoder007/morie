# morie.fn -- function file (rootcoder007/morie)
"""LAMB -- layer-wise adaptive moments, You et al. (2020)."""

from __future__ import annotations

from . import _array_core as np

from ._optim import as_vector, init_state, step_result

__all__ = ["lamb_optimizer"]


def lamb_optimizer(g, w, lr=0.001, beta1=0.9, beta2=0.999, wd=0.01, eps=1e-6, state=None):
    r"""One LAMB step for a single layer's parameters.

    LAMB is LARS's trust ratio applied to Adam's direction rather than to the
    raw gradient. Form the bias-corrected Adam direction

    .. math::
        r_t = \hat m_t / (\sqrt{\hat v_t} + \epsilon) + \lambda w_t,

    then scale it by the layer trust ratio
    :math:`\phi = \lVert w \rVert / \lVert r_t \rVert`:

    .. math::
        \Delta w_t = -\,\alpha\, \phi\, r_t .

    Because :math:`\lVert r_t \rVert` already sits near 1 per coordinate for
    Adam, :math:`\phi` is close to :math:`\lVert w \rVert / \sqrt{p}`, which
    is what lets LAMB hold up at batch sizes where Adam alone diverges.

    As in LARS, a zero weight or direction norm falls back to :math:`\phi=1`.

    Parameters
    ----------
    g : array-like
        Gradient for this layer.
    w : array-like
        Current weights for this layer; same size as ``g``.
    lr : float
        Global learning rate.
    beta1, beta2 : float
        Moment decay rates, in [0, 1).
    wd : float
        Weight decay, applied to the Adam direction (as in AdamW).
    eps : float
        Denominator floor.
    state : dict, optional
        ``state`` from the previous call.

    Returns
    -------
    RichResult
        ``update``, ``state``, ``trust_ratio``, ``m``, ``v``.

    References
    ----------
    You, Y., Li, J., Reddi, S., et al. (2020). Large batch optimization for
        deep learning: Training BERT in 76 minutes. *ICLR 2020*.
        arXiv:1904.00962.

    Examples
    --------
    Step length scales with the weight norm, as for LARS.

    >>> import numpy as np
    >>> a = lamb_optimizer([1.0, 1.0], [1.0, 1.0], lr=1.0, wd=0.0)["update"]
    >>> b = lamb_optimizer([1.0, 1.0], [2.0, 2.0], lr=1.0, wd=0.0)["update"]
    >>> bool(abs(np.linalg.norm(b) - 2 * np.linalg.norm(a)) < 1e-9)
    True

    Zero weights fall back to an untrusted step.

    >>> float(lamb_optimizer([1.0], [0.0], lr=0.1, wd=0.0)["trust_ratio"])
    1.0
    """
    if not 0.0 <= beta1 < 1.0:
        raise ValueError("beta1 must be in [0, 1)")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must be in [0, 1)")
    g = as_vector(g)
    w = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    if w.size != g.size:
        raise ValueError(f"w has {w.size} entries but g has {g.size}")
    st = init_state(state, g.size, keys=("m", "v"))
    t = st["t"]
    st["m"] = beta1 * st["m"] + (1.0 - beta1) * g
    st["v"] = beta2 * st["v"] + (1.0 - beta2) * g**2
    m_hat = st["m"] / (1.0 - beta1**t)
    v_hat = st["v"] / (1.0 - beta2**t)
    r = m_hat / (np.sqrt(v_hat) + eps) + wd * w
    wn = float(np.linalg.norm(w))
    rn = float(np.linalg.norm(r))
    trust = wn / rn if wn > 0 and rn > 0 else 1.0
    update = -lr * trust * r
    return step_result(
        update, st, "LAMB", trust_ratio=float(trust), w_norm=wn, r_norm=rn,
        m=st["m"], v=st["v"], direction=r,
    )


def cheatsheet():
    return "lambopt: LAMB -- LARS trust ratio applied to Adam's direction; scales to very large batches"
