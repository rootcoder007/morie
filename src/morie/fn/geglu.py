# morie.fn -- slice s03 (rootcoder007/morie)
"""GEGLU gated activation.

Source consulted (FETCHED): Shazeer, N. (2020).  GLU variants improve
transformer.  arXiv:2002.05202:

    GEGLU(x, W, V, b, c) = GELU(xW + b) (x) (xV + c)

with GELU(z) = z Phi(z), the *exact* Gaussian error linear unit of
Hendrycks and Gimpel (2016), arXiv:1606.08415 -- the tanh expression
that circulates as "GELU" is an approximation to it, and is not used
here, because at 1e-9 the two differ.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["geglu_activation"]


def geglu_activation(y, x=None, W=None, V=None, b=None, c=None, W2=None):
    """GEGLU(x, W, V, b, c), and optionally the full FFN.

    Returns
    -------
    RichResult with payload:
        estimate : the first output unit
        out, gate, ffn
    """
    v = k.vec(x if x is not None else y)
    g = k.matvec(k.tr(k.mat(W)), v)
    u = k.matvec(k.tr(k.mat(V)), v)
    bb = k.vec(b) if b is not None else [0.0] * len(g)
    cc = k.vec(c) if c is not None else [0.0] * len(u)
    gate = [k.gelu(g[i] + bb[i]) for i in range(len(g))]
    out = [gate[i] * (u[i] + cc[i]) for i in range(len(g))]
    ffn = k.matvec(k.tr(k.mat(W2)), out) if W2 is not None else []
    return RichResult(
        title="GEGLU",
        summary_lines=[("units", len(out))],
        payload={
            "estimate": out[0] if out else float("nan"),
            "out": out,
            "gate": gate,
            "ffn": ffn,
            "method": "GEGLU gated activation with the exact GELU (Shazeer 2020)",
        },
    )


def cheatsheet():
    return "geglu: GEGLU gated activation"
