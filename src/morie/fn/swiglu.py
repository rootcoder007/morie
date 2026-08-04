# morie.fn -- slice s03 (rootcoder007/morie)
"""SwiGLU gated activation.

Source consulted (FETCHED): Shazeer, N. (2020).  GLU variants improve
transformer.  arXiv:2002.05202, which prints the family verbatim:

    ReGLU(x, W, V, b, c)        = max(0, xW + b)  (x) (xV + c)
    GEGLU(x, W, V, b, c)        = GELU(xW + b)    (x) (xV + c)
    SwiGLU(x, W, V, b, c, beta) = Swish_beta(xW + b) (x) (xV + c)

with (x) the elementwise product, Swish_beta(z) = z sigma(beta z)
(Ramachandran et al. 2017) and GELU(z) = z Phi(z) (Hendrycks and Gimpel
2016), both also quoted in the paper.  The feed-forward layer built from
it, the paper's FFN_SwiGLU(x, W, V, W2) = (Swish_1(xW) (x) xV) W2,
"reduce[s] the number of hidden units ... by a factor of 2/3" so that
the parameter count matches the ordinary two-matrix FFN -- that
adjustment is *not* applied here, because the caller supplies W and V
and so has already fixed the width.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["swiglu_activation"]


def swiglu_activation(y, x=None, W=None, V=None, b=None, c=None, beta=1.0,
                      W2=None):
    """SwiGLU(x, W, V, b, c, beta), and optionally the full FFN.

    Parameters
    ----------
    y : array-like
        The input x.  (First slot, for signature stability; when ``x`` is
        also given, ``x`` wins.)
    x : array-like, optional
        The input.
    W, V : 2-D array-like
        The gate and value projections, rows indexed by input unit.
    b, c : array-like, optional
        Biases; zero by default.
    beta : float
        The Swish parameter; 1 in the paper's FFN.
    W2 : 2-D array-like, optional
        Output projection; when given, the FFN output is returned too.

    Returns
    -------
    RichResult with payload:
        estimate : the first output unit
        out      : the SwiGLU vector
        gate     : Swish_beta(xW + b)
        ffn      : (SwiGLU) W2, empty when W2 is None
    """
    v = k.vec(x if x is not None else y)
    Wm = k.mat(W)
    Vm = k.mat(V)
    g = k.matvec(k.tr(Wm), v)
    u = k.matvec(k.tr(Vm), v)
    bb = k.vec(b) if b is not None else [0.0] * len(g)
    cc = k.vec(c) if c is not None else [0.0] * len(u)
    gate = [k.swish(g[i] + bb[i], float(beta)) for i in range(len(g))]
    out = [gate[i] * (u[i] + cc[i]) for i in range(len(g))]
    ffn = k.matvec(k.tr(k.mat(W2)), out) if W2 is not None else []
    return RichResult(
        title="SwiGLU",
        summary_lines=[("units", len(out))],
        payload={
            "estimate": out[0] if out else float("nan"),
            "out": out,
            "gate": gate,
            "ffn": ffn,
            "beta": float(beta),
            "method": "SwiGLU gated activation (Shazeer 2020)",
        },
    )


def cheatsheet():
    return "swiglu: SwiGLU gated activation (used in PaLM, LLaMA)"
