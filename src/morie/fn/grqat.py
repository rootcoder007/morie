# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantization-aware training: fake-quantize forward, straight-through backward."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_quantization_aware_training"]

_METHOD = "Quantization-aware training (fake quant + STE)"


def geron_quantization_aware_training(x, s, bits=8, upstream_grad=None):
    r"""Simulate quantization in the forward pass, pass the gradient through.

    .. math::
        y = \mathrm{dequant}(\mathrm{quant}(x))
          = s \cdot \mathrm{clip}\bigl(\mathrm{round}(x/s), -q_{\max}, q_{\max}\bigr)

    and on the way back the straight-through estimator uses
    :math:`\partial y/\partial x = 1` inside the representable range and
    0 outside.  The true derivative of ``round`` is zero almost
    everywhere, so honest backprop would deliver no gradient at all --
    STE is the deliberate lie that makes QAT trainable.  Clipping is the
    part that is *not* a lie: a value pushed past :math:`q_{\max}`
    genuinely cannot move the output, so its gradient is genuinely zero,
    and reporting the clipped fraction is how you notice a badly chosen
    scale.

    Parameters
    ----------
    x : array-like
    s : float
        Positive quantization step.
    bits : int, optional
    upstream_grad : array-like, optional
        Gradient arriving from the next layer; defaults to ones.

    Returns
    -------
    RichResult
        Payload keys ``y`` (fake-quantized), ``q``, ``ste_mask``,
        ``grad_x``, ``clipped_fraction``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Appendix B, Quantization-Aware Training section.

    Examples
    --------
    Step 0.1: 0.24 snaps to 0.2 and its gradient passes through; 20.0 is
    far past ``127 * 0.1 = 12.7`` so it clips and its gradient is killed.

    >>> r = geron_quantization_aware_training([0.24, 20.0], s=0.1)
    >>> [round(v, 10) for v in r["y"]]
    [0.2, 12.7]
    >>> r["ste_mask"]
    [1.0, 0.0]
    >>> r["clipped_fraction"]
    0.5

    Inside the range the layer is transparent to the gradient:

    >>> geron_quantization_aware_training([0.24], s=0.1, upstream_grad=[3.0])["grad_x"]
    [3.0]
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("x is empty.")
    if not np.all(np.isfinite(a)):
        raise ValueError("x contains non-finite values.")
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"s must be a positive finite step, got {s}.")
    bits = int(bits)
    if bits < 2 or bits > 32:
        raise ValueError(f"bits must lie in [2, 32], got {bits}.")
    qmax = 2 ** (bits - 1) - 1

    raw = np.round(a / s)
    q = np.clip(raw, -qmax, qmax)
    y = q * s
    mask = (np.abs(a / s) <= qmax).astype(float)
    if upstream_grad is None:
        g = np.ones_like(a)
    else:
        g = np.asarray(upstream_grad, dtype=float)
        if g.shape != a.shape:
            raise ValueError(f"upstream_grad shape {g.shape} != x shape {a.shape}.")
        if not np.all(np.isfinite(g)):
            raise ValueError("upstream_grad contains non-finite values.")
    grad = g * mask

    return RichResult(
        title="Quantization-aware training",
        summary_lines=[("Step", s), ("Clipped fraction", float(1.0 - mask.mean()))],
        payload={
            "y": y.tolist(),
            "q": q.tolist(),
            "ste_mask": mask.tolist(),
            "grad_x": grad.tolist(),
            "clipped_fraction": float(1.0 - mask.mean()),
            "estimate": y.tolist(),
            "n": int(a.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grqat: y = s*clip(round(x/s)); STE passes grad through in-range, kills it where clipped"
