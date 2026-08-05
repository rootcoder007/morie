# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""INT8 quantised attention with per-row scales.

Source: Dettmers, T., Lewis, M., Belkada, Y. and Zettlemoyer, L. (2022),
"LLM.int8(): 8-bit matrix multiplication for transformers at scale",
NeurIPS 2022, arXiv:2208.07339, read from the fetched PDF.  The paper's
first ingredient is vector-wise quantisation: rather than one scale for
a whole tensor, each row of the left operand and each column of the
right operand gets its own absmax scale, the product is accumulated in
int32, and dequantisation divides by the outer product of the two
scales.  For a row q_i and a key k_j,

    s_q(i) = max_t |Q_it| / 127,   s_k(j) = max_t |K_jt| / 127,
    Q_int  = round(Q / s_q),       K_int  = round(K / s_k),
    (Q K^T)_ij  ~=  s_q(i) s_k(j) * (Q_int K_int^T)_ij,

and the same treatment is applied to the value matmul.  Softmax itself
is done in floating point: quantising the probabilities is what destroys
the method, since they span several orders of magnitude within a row.

Why per-row and not per-tensor: a single outlier feature forces one
global scale to be huge, and every other entry then quantises to zero.
That is the failure the paper is about.  The anchor exercises it
directly -- a matrix with one large row is quantised, dequantised and
compared per row, and the small rows must survive.

The exactness anchor is the other one: when every entry is already an
exact multiple of its row scale, rounding does nothing and the output
must equal float attention to machine precision.  Rounding is
half-away-from-zero in both language arms, which is R's convention under
an explicit floor(x + 0.5) for positive values, chosen because R's
round() is half-to-even and Python's is too, but neither is the
hardware convention here -- pinning it explicitly keeps the two arms
bit-identical.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["int8_attention"]

QMAX = 127.0


def round_half_away(x):
    """Half-away-from-zero rounding, pinned so both arms agree bitwise."""
    return math.floor(x + 0.5) if x >= 0.0 else -math.floor(-x + 0.5)


def row_scales(M):
    """Per-row absmax scale; a row of zeros gets scale 1 so nothing divides by 0."""
    out = []
    for r in M:
        a = 0.0
        for v in r:
            if abs(v) > a:
                a = abs(v)
        out.append(a / QMAX if a > 0.0 else 1.0)
    return out


def quantise(M, s):
    """Row-wise quantisation to the int8 lattice, clamped to [-127, 127]."""
    out = []
    for i, r in enumerate(M):
        row = []
        for v in r:
            q = round_half_away(v / s[i])
            if q > QMAX:
                q = QMAX
            if q < -QMAX:
                q = -QMAX
            row.append(float(q))
        out.append(row)
    return out


def int8_attention(y=None, Q=None, K=None, V=None, scales=None):
    """Attention computed through an int8 lattice.

    Parameters
    ----------
    y : ignored
        Accepted for interface compatibility with the rest of the shelf.
    Q : array-like
        n_q by d queries.
    K : array-like
        n_k by d keys.
    V : array-like
        n_k by d_v values.
    scales : sequence of three sequences, optional
        Explicit per-row scales for Q, K and V; absmax scales by default.

    Returns
    -------
    output : the n_q by d_v attention output
    scores : the dequantised pre-softmax scores
    s_q, s_k, s_v : the scales actually used
    max_abs_error_vs_float : how far the int8 path fell from exact attention
    """
    if Q is None or K is None or V is None:
        raise ValueError("int8_attention: Q, K and V are all required")
    Qm = [[float(v) for v in r] for r in Q]
    Km = [[float(v) for v in r] for r in K]
    Vm = [[float(v) for v in r] for r in V]
    nq = len(Qm)
    nk = len(Km)
    if nq == 0 or nk == 0:
        raise ValueError("int8_attention: Q and K must be non-empty")
    d = len(Qm[0])
    if any(len(r) != d for r in Qm) or any(len(r) != d for r in Km):
        raise ValueError("int8_attention: Q and K must share the key dimension")
    if len(Vm) != nk:
        raise ValueError("int8_attention: V must have one row per key")
    dv = len(Vm[0])
    if scales is None:
        sq = row_scales(Qm)
        sk = row_scales(Km)
        sv = row_scales(Vm)
    else:
        sl = list(scales)
        if len(sl) != 3:
            raise ValueError("int8_attention: scales must hold three vectors, for Q, K and V")
        sq = [float(v) for v in sl[0]]
        sk = [float(v) for v in sl[1]]
        sv = [float(v) for v in sl[2]]
        if len(sq) != nq or len(sk) != nk or len(sv) != nk:
            raise ValueError("int8_attention: a scale vector has the wrong length")
        for v in sq + sk + sv:
            if not (v > 0.0):
                raise ValueError("int8_attention: scales must be positive")
    Qi = quantise(Qm, sq)
    Ki = quantise(Km, sk)
    Vi = quantise(Vm, sv)
    sc = 1.0 / math.sqrt(d)
    S = []
    for i in range(nq):
        row = []
        for j in range(nk):
            acc = 0.0
            for t in range(d):
                acc += Qi[i][t] * Ki[j][t]
            row.append(acc * sq[i] * sk[j] * sc)
        S.append(row)
    O = []
    W = []
    for i in range(nq):
        mx = max(S[i])
        e = [math.exp(v - mx) for v in S[i]]
        tot = 0.0
        for v in e:
            tot += v
        w = [v / tot for v in e]
        W.append(w)
        o = [0.0] * dv
        for j in range(nk):
            wj = w[j] * sv[j]
            for t in range(dv):
                o[t] += wj * Vi[j][t]
        O.append(o)
    # exact float attention, for the error report only
    err = 0.0
    for i in range(nq):
        row = []
        for j in range(nk):
            acc = 0.0
            for t in range(d):
                acc += Qm[i][t] * Km[j][t]
            row.append(acc * sc)
        mx = max(row)
        e = [math.exp(v - mx) for v in row]
        tot = 0.0
        for v in e:
            tot += v
        for t in range(dv):
            ref = 0.0
            for j in range(nk):
                ref += (e[j] / tot) * Vm[j][t]
            dd = abs(O[i][t] - ref)
            if dd > err:
                err = dd
    return RichResult(
        title="INT8 quantised attention",
        summary_lines=[("n_q", nq), ("n_k", nk), ("max_abs_error", err)],
        payload={
            "output": O,
            "estimate": O[0][0],
            "scores": S,
            "weights": W,
            "s_q": sq,
            "s_k": sk,
            "s_v": sv,
            "max_abs_error_vs_float": err,
            "n_q": nq,
            "n_k": nk,
            "d": d,
            "d_v": dv,
            "method": "vector-wise int8 quantisation, float softmax; Dettmers et al. (2022), arXiv:2208.07339",
        },
    )


def cheatsheet():
    return "atq8: INT8 quantised attention via per-row scales"


# compact alias per ledger/NAMING.md
int8attention = int8_attention
