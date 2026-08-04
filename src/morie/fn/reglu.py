# morie.fn -- slice s03 (rootcoder007/morie)
"""REGLU gated activation.

Source consulted (FETCHED): Shazeer, N. (2020).  GLU variants improve
transformer.  arXiv:2002.05202:

    ReGLU(x, W, V, b, c) = max(0, xW + b) (x) (xV + c)

the rectifier in place of the sigmoid of Dauphin et al.'s (2017)
original gated linear unit.  Because the gate is exactly zero on half
its domain, the count of dead units is reported: it is the diagnostic
that distinguishes ReGLU from its smooth siblings.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["reglu_activation"]


def reglu_activation(y, x=None, W=None, V=None, b=None, c=None, W2=None):
    """ReGLU(x, W, V, b, c), and optionally the full FFN.

    Returns
    -------
    RichResult with payload:
        estimate : the first output unit
        out, gate, ffn
        n_dead   : units whose gate is exactly zero
    """
    v = k.vec(x if x is not None else y)
    g = k.matvec(k.tr(k.mat(W)), v)
    u = k.matvec(k.tr(k.mat(V)), v)
    bb = k.vec(b) if b is not None else [0.0] * len(g)
    cc = k.vec(c) if c is not None else [0.0] * len(u)
    gate = [k.relu(g[i] + bb[i]) for i in range(len(g))]
    out = [gate[i] * (u[i] + cc[i]) for i in range(len(g))]
    dead = 0
    for z in gate:
        if z == 0.0:
            dead += 1
    ffn = k.matvec(k.tr(k.mat(W2)), out) if W2 is not None else []
    return RichResult(
        title="ReGLU",
        summary_lines=[("units", len(out)), ("dead gates", dead)],
        payload={
            "estimate": out[0] if out else float("nan"),
            "out": out,
            "gate": gate,
            "ffn": ffn,
            "n_dead": dead,
            "method": "ReGLU gated activation (Shazeer 2020)",
        },
    )


def cheatsheet():
    return "reglu: REGLU gated activation"
