# morie.fn -- slice s03 (rootcoder007/morie)
"""One LLaMA decoder block.

Source consulted (FETCHED): Touvron, H. et al. (2023).  LLaMA: open and
efficient foundation language models.  arXiv:2302.13971, section 2.2,
which lists the three departures from the original transformer:

* pre-normalisation -- "we normalize the input of each transformer
  sub-layer, instead of normalizing the output", using RMSNorm (Zhang
  and Sennrich 2019);
* the SwiGLU activation of Shazeer (2020) in place of ReLU, with hidden
  dimension 2/3 * 4d rather than 4d;
* rotary positional embeddings (Su et al. 2021, arXiv:2104.09864) in
  place of absolute positions.

So the block computed here is

    h = x + Attn(RMSNorm(x))
    y = h + SwiGLU-FFN(RMSNorm(h))

with RoPE applied to the queries and keys inside the attention.  RoPE
rotates each consecutive pair of coordinates by m theta_j with
theta_j = 10000^(-2j/d), which is the form Su et al. print and which
Touvron et al. adopt unchanged.

Attention here is single-head and causal; the caller supplies the
projections, so nothing is invented.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["llama"]


def _rope(v, pos, base=10000.0):
    d = len(v)
    out = list(v)
    for j in range(0, d - 1, 2):
        th = pos * (base ** (-float(j) / d))
        c = math.cos(th)
        s = math.sin(th)
        a = v[j]
        b = v[j + 1]
        out[j] = a * c - b * s
        out[j + 1] = a * s + b * c
    return out


def llama(tokens, model=None, Wq=None, Wk=None, Wv=None, Wo=None,
          W1=None, W3=None, W2=None, g1=None, g2=None, rope_base=10000.0):
    """A single pre-norm, RoPE, SwiGLU decoder block over a token sequence.

    Parameters
    ----------
    tokens : 2-D array-like
        Token embeddings, one row per position.
    model : dict, optional
        Alternative container for the projections, keyed by the argument
        names below.
    Wq, Wk, Wv, Wo : 2-D array-like
        Attention projections (input units in rows).
    W1, W3, W2 : 2-D array-like
        SwiGLU gate, value and output projections.
    g1, g2 : array-like, optional
        RMSNorm gains for the two sub-layers.
    rope_base : float
        The RoPE base; 10000 in the paper.

    Returns
    -------
    RichResult with payload:
        estimate : y[last][0]
        out      : the block output, one row per position
        attn     : the causal attention weights
        h        : the post-attention residual stream
    """
    if model is not None:
        Wq = model.get("Wq", Wq)
        Wk = model.get("Wk", Wk)
        Wv = model.get("Wv", Wv)
        Wo = model.get("Wo", Wo)
        W1 = model.get("W1", W1)
        W3 = model.get("W3", W3)
        W2 = model.get("W2", W2)
    X = k.mat(tokens)
    n = len(X)
    d = len(X[0]) if n else 0

    def rms(v, gain):
        s = 0.0
        for z in v:
            s += z * z
        r = math.sqrt(s / len(v)) if v else 0.0
        gg = k.vec(gain) if gain is not None else [1.0] * len(v)
        return [(v[i] / r) * gg[i] if r > 0.0 else 0.0 for i in range(len(v))]

    Qm = k.tr(k.mat(Wq))
    Km = k.tr(k.mat(Wk))
    Vm = k.tr(k.mat(Wv))
    q = []
    kk = []
    vv = []
    for t in range(n):
        xn = rms(X[t], g1)
        q.append(_rope(k.matvec(Qm, xn), t, rope_base))
        kk.append(_rope(k.matvec(Km, xn), t, rope_base))
        vv.append(k.matvec(Vm, xn))
    dk = len(q[0]) if q else 1
    attn = []
    ctx = []
    for t in range(n):
        logits = []
        for m in range(t + 1):
            s = 0.0
            for a in range(dk):
                s += q[t][a] * kk[m][a]
            logits.append(s / math.sqrt(dk))
        w = k.softmax(logits)
        attn.append(w)
        c = [0.0] * len(vv[0])
        for m in range(t + 1):
            for b in range(len(c)):
                c[b] += w[m] * vv[m][b]
        ctx.append(c)
    Om = k.tr(k.mat(Wo))
    h = []
    for t in range(n):
        o = k.matvec(Om, ctx[t])
        h.append([X[t][j] + o[j] for j in range(d)])
    A = k.tr(k.mat(W1))
    B = k.tr(k.mat(W3))
    C = k.tr(k.mat(W2))
    out = []
    for t in range(n):
        hn = rms(h[t], g2)
        gt = k.matvec(A, hn)
        ut = k.matvec(B, hn)
        mid = [k.swish(gt[i]) * ut[i] for i in range(len(gt))]
        o = k.matvec(C, mid)
        out.append([h[t][j] + o[j] for j in range(d)])
    return RichResult(
        title="LLaMA decoder block",
        summary_lines=[("positions", n), ("d_model", d)],
        payload={
            "estimate": out[-1][0] if out else float("nan"),
            "out": out,
            "attn": attn,
            "h": h,
            "method": "LLaMA block: pre-RMSNorm, RoPE attention, SwiGLU FFN (Touvron et al. 2023 sec. 2.2)",
        },
    )


def cheatsheet():
    return "llama: LLaMA decoder (RMSNorm, RoPE, SwiGLU)"
