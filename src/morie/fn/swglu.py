"""SwiGLU activation."""

from __future__ import annotations

from ._containers import DescriptiveResult
from .swigl import swiglu_activation


def swiglu(x, W1=None, W2=None, W3=None) -> DescriptiveResult:
    r"""SwiGLU feed-forward block ``(x W1 * SiLU(x W3)) W2`` (Shazeer 2020).

    The gate ``SiLU(x W3) = x W3 sigma(x W3)`` multiplies the up projection
    ``x W1`` elementwise and ``W2`` projects back; without ``W1``/``W3`` the
    gating is elementwise (``x * SiLU(x)``) and without ``W2`` the hidden
    layer is returned. Thin front-end to
    :func:`morie.fn.swigl.swiglu_activation` (``W = W3``, ``V = W1``);
    ``value`` is the output array, ``extra["hidden"]`` the gated layer.

    References
    ----------
    Shazeer, N. (2020). GLU variants improve Transformer. arXiv:2002.05202.

    Examples
    --------
    >>> round(float(swiglu([[1.0, -2.0]]).value[0][0]), 12)
    0.73105857863
    """
    if (W1 is None) != (W3 is None):
        raise ValueError("provide both W1 and W3 or neither")
    h = swiglu_activation(x, W3, W1)["tensor"]
    out = h if W2 is None else h @ W2
    return DescriptiveResult(name="swiglu", value=out, extra={"output": out, "hidden": h})


def cheatsheet() -> str:
    return "swiglu(x, W1, W2, W3) -> (x W1 * SiLU(x W3)) W2 (Shazeer 2020)."


swglu = swiglu
