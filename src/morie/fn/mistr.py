# morie.fn -- function file (rootcoder007/morie)
r"""Mistral: sliding-window attention, GQA, RoPE, SwiGLU, RMSNorm.

Four substitutions to a standard decoder block, each with a property
that can be checked exactly rather than taken on faith.

**Sliding-window attention.** Each token attends only to the previous
:math:`W`. That sounds like a hard horizon and is not: attention
composes across layers, so after :math:`k` layers information has
travelled up to :math:`kW` tokens. The paper's span at
:math:`W = 4096` over 32 layers is roughly 131k. What the window
actually buys is a cache bounded by :math:`W` instead of by the
sequence.

**Rotary position embedding.** Position enters by rotating each pair of
channels by an angle proportional to the index,

.. math:: f(x, m) = R_m x, \qquad
          R_m = \begin{pmatrix}\cos m\theta & -\sin m\theta\\
          \sin m\theta & \cos m\theta\end{pmatrix},

and the identity that matters is
:math:`\langle R_m q, R_n k\rangle = \langle R_{m-n} q, k\rangle`: the
score depends on :math:`m - n` alone. That is exact, holds for every
:math:`m` and :math:`n`, and is what the anchor checks -- an
implementation that rotated by the wrong sign or applied the rotation
to the wrong channel pairs still runs and still trains, just without
the property it was chosen for.

**Grouped-query attention.** Several query heads share one key-value
head, cutting the cache by the sharing factor. With one group it is
multi-query attention and with as many groups as heads it is ordinary
multi-head attention; both are reachable here, and both are checked
against the general path.

**SwiGLU and RMSNorm.** :math:`\mathrm{SwiGLU}(x) =
(\mathrm{Swish}(xW_1) \odot xW_3)W_2` -- a gate, not an activation, so
half the projection controls the other half. RMSNorm drops the
mean-centring of LayerNorm and keeps only the scale, which makes it
exactly invariant to the *scale* of its input and, unlike LayerNorm,
NOT invariant to a shift. Both properties are checked because they are
what distinguishes it.

References
----------
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S.,
de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L.,
Lavaud, L. R., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T.,
Wang, T., Lacroix, T. & El Sayed, W. (2023) "Mistral 7B",
arXiv:2310.06825. Sliding-window attention, GQA, and the architecture
summarised here.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W. & Liu, Y. (2024) "RoFormer:
Enhanced Transformer with Rotary Position Embedding",
*Neurocomputing* 568, 127063, doi:10.1016/j.neucom.2023.127063,
arXiv:2104.09864. The rotation and its relative-position property.

Shazeer, N. (2020) "GLU Variants Improve Transformer",
arXiv:2002.05202. SwiGLU.

Zhang, B. & Sennrich, R. (2019) "Root Mean Square Layer
Normalization", *Advances in Neural Information Processing Systems* 32,
arXiv:1910.07467. RMSNorm.

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F. &
Sanghai, S. (2023) "GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints", *Proceedings of EMNLP 2023*,
arXiv:2305.13245. Grouped-query attention.

Beltagy, I., Peters, M. E. & Cohan, A. (2020) "Longformer: The
Long-Document Transformer", arXiv:2004.05150. The sliding-window
attention pattern Mistral adopts.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["rope_angles", "apply_rope", "sliding_window_mask",
           "grouped_query_attention", "swiglu", "rms_norm",
           "mistral_block", "attention_span"]

_EPS = 1e-12


def rms_norm(x, weight=None, eps=1e-6):
    r""":math:`x / \sqrt{\mathrm{mean}(x^2) + \epsilon}`, times a gain.

    No mean subtraction: RMSNorm is invariant to the SCALE of its input
    and, unlike LayerNorm, not to a shift. The invariance is exact only
    at ``eps = 0``; any positive eps breaks it slightly, by design --
    the anchor checks it at eps = 0 and separately measures what a
    realistic eps costs, rather than asserting an exactness the
    implementation does not have.
    """
    d = len(x)
    if d == 0:
        raise ValueError("mistr: empty vector")
    ms = sum(v * v for v in x) / d
    inv = 1.0 / math.sqrt(ms + eps)
    if weight is None:
        return [v * inv for v in x]
    if len(weight) != d:
        raise ValueError("mistr: gain has %d entries for %d channels"
                         % (len(weight), d))
    return [x[i] * inv * weight[i] for i in range(d)]


def swiglu(x, W1, W2, W3):
    r""":math:`(\mathrm{Swish}(xW_1) \odot xW_3)\,W_2`.

    A gate rather than an activation: the :math:`W_3` branch multiplies
    the :math:`W_1` branch elementwise, so one half decides how much of
    the other half survives.
    """
    a = [sum(x[i] * W1[i][j] for i in range(len(x)))
         for j in range(len(W1[0]))]
    b = [sum(x[i] * W3[i][j] for i in range(len(x)))
         for j in range(len(W3[0]))]
    if len(a) != len(b):
        raise ValueError("mistr: W1 and W3 must have the same width")
    gated = [k.sigmoid(a[j]) * a[j] * b[j] for j in range(len(a))]
    return [sum(gated[j] * W2[j][c] for j in range(len(gated)))
            for c in range(len(W2[0]))]


def rope_angles(d, base=10000.0):
    r""":math:`\theta_i = \mathrm{base}^{-2i/d}` for each channel pair."""
    if d % 2 != 0:
        raise ValueError("mistr: RoPE needs an even dimension, got %d"
                         % d)
    return [base ** (-2.0 * i / d) for i in range(d // 2)]


def apply_rope(x, pos, theta=None, base=10000.0):
    r"""Rotate each channel PAIR by ``pos`` times its angle.

    Pairs are :math:`(2i, 2i+1)`. Rotating the wrong pairing -- for
    instance the first half against the second -- is a common variant
    and breaks the relative-position identity unless the angles are
    permuted to match, which is why the pairing is spelled out.
    """
    d = len(x)
    th = rope_angles(d, base) if theta is None else list(theta)
    if len(th) != d // 2:
        raise ValueError("mistr: %d angles for %d channels"
                         % (len(th), d))
    out = [0.0] * d
    for i in range(d // 2):
        ang = pos * th[i]
        c, s = math.cos(ang), math.sin(ang)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out


def sliding_window_mask(L, window, causal=True):
    r"""Row :math:`i` may attend to :math:`j` when
    :math:`i - W < j \le i`."""
    if window < 1:
        raise ValueError("mistr: window must be at least 1, got %d"
                         % window)
    mask = []
    for i in range(L):
        row = []
        for j in range(L):
            ok = (j <= i or not causal) and (i - j) < window
            row.append(bool(ok))
        mask.append(row)
    return mask


def attention_span(window, n_layers):
    """Theoretical span: attention composes across layers, so a window
    of W over k layers reaches about k*W tokens."""
    return int(window) * int(n_layers)


def grouped_query_attention(Q, K, V, n_heads, n_kv_heads, mask=None,
                            positions=None, base=10000.0):
    r"""Attention with :math:`n_{kv}` key-value heads shared across
    :math:`n_h` query heads.

    ``n_kv_heads == n_heads`` is ordinary multi-head attention and
    ``n_kv_heads == 1`` is multi-query attention; both are reachable so
    the general path can be checked against them.
    """
    Qm, Km, Vm = k.mat(Q), k.mat(K), k.mat(V)
    L = len(Qm)
    if len(Km) != L or len(Vm) != L:
        raise ValueError("mistr: Q, K and V must have the same length")
    d = len(Qm[0])
    if n_heads < 1 or n_kv_heads < 1:
        raise ValueError("mistr: need at least one head of each kind")
    if n_heads % n_kv_heads != 0:
        raise ValueError("mistr: n_heads (%d) must be a multiple of "
                         "n_kv_heads (%d)" % (n_heads, n_kv_heads))
    if d % n_heads != 0:
        raise ValueError("mistr: dimension %d is not divisible by %d "
                         "heads" % (d, n_heads))
    hd = d // n_heads
    # K and V carry n_kv_heads heads of the SAME head dimension as Q --
    # that is where the cache saving comes from, so the width is
    # n_kv_heads*hd and not d.
    dk = len(Km[0])
    if dk != n_kv_heads * hd:
        raise ValueError("mistr: K and V must be %d wide (n_kv_heads=%d "
                         "times head_dim=%d), got %d"
                         % (n_kv_heads * hd, n_kv_heads, hd, dk))
    kd = hd
    group = n_heads // n_kv_heads
    pos = list(range(L)) if positions is None else list(positions)
    out = [[0.0] * d for _ in range(L)]
    for h in range(n_heads):
        g = h // group
        qs = [Qm[t][h * hd:(h + 1) * hd] for t in range(L)]
        ks = [Km[t][g * kd:(g + 1) * kd] for t in range(L)]
        vs = [Vm[t][g * kd:(g + 1) * kd] for t in range(L)]
        if positions is not False:
            qs = [apply_rope(qs[t], pos[t], base=base) for t in range(L)]
            ks = [apply_rope(ks[t], pos[t], base=base) for t in range(L)]
        scale = 1.0 / math.sqrt(hd)
        for i in range(L):
            allowed = [j for j in range(L)
                       if mask is None or mask[i][j]]
            if not allowed:
                raise ValueError("mistr: row %d may attend to nothing"
                                 % i)
            sc = [scale * sum(qs[i][c] * ks[j][c] for c in range(hd))
                  for j in allowed]
            mx = max(sc)
            w = [math.exp(v - mx) for v in sc]
            tot = sum(w)
            for c in range(hd):
                out[i][h * hd + c] = sum(
                    w[t] * vs[allowed[t]][c]
                    for t in range(len(allowed))) / tot
    return out


def mistral_block(X, Wq, Wk, Wv, Wo, W1, W2, W3, n_heads, n_kv_heads,
                  window, norm1=None, norm2=None, base=10000.0):
    """One decoder block: RMSNorm, SWA + GQA + RoPE, residual, RMSNorm,
    SwiGLU, residual."""
    Xm = k.mat(X)
    L = len(Xm)
    d = len(Xm[0])
    mask = sliding_window_mask(L, window)

    def proj(row, Wm):
        return [sum(row[i] * Wm[i][j] for i in range(len(row)))
                for j in range(len(Wm[0]))]

    h = [rms_norm(Xm[t], norm1) for t in range(L)]
    Q = [proj(h[t], Wq) for t in range(L)]
    K = [proj(h[t], Wk) for t in range(L)]
    V = [proj(h[t], Wv) for t in range(L)]
    a = grouped_query_attention(Q, K, V, n_heads, n_kv_heads, mask=mask,
                                base=base)
    a = [proj(a[t], Wo) for t in range(L)]
    x1 = [[Xm[t][c] + a[t][c] for c in range(d)] for t in range(L)]
    h2 = [rms_norm(x1[t], norm2) for t in range(L)]
    f = [swiglu(h2[t], W1, W2, W3) for t in range(L)]
    out = [[x1[t][c] + f[t][c] for c in range(d)] for t in range(L)]
    return RichResult(payload={
        "estimate": out, "output": out, "attention_mask": mask,
        "L": L, "d": d, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "window": int(window),
        "kv_cache_entries": min(int(window), L) * n_kv_heads,
        "method": "Mistral decoder block: SWA + GQA + RoPE + SwiGLU + "
                  "RMSNorm, Jiang et al. (2023)",
    })


def cheatsheet():
    return ("mistr: SWA -- token i attends to (i-W, i]; span grows to "
            "k*W over k layers because attention composes. RoPE rotates "
            "pairs (2i, 2i+1) by pos*theta_i, and <R_m q, R_n k> = "
            "<R_{m-n} q, k> EXACTLY. GQA shares one kv head across "
            "n_heads/n_kv query heads. SwiGLU gates; RMSNorm is "
            "scale-invariant but NOT shift-invariant.")


# compact alias per ledger/NAMING.md
mistralblock = mistral_block
