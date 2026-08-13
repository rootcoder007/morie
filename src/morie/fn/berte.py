# morie.fn -- function file (rootcoder007/morie)
r"""BERT: the bidirectional encoder forward pass.

An encoder block, repeated: multi-head self-attention over the whole
sequence in both directions, then a position-wise feed-forward network,
each wrapped in a residual connection and LayerNorm.

**Bidirectional is the whole claim, and it is why the training task had
to change.** A left-to-right model can be trained to predict the next
token; a model that sees both sides cannot, because the answer is in the
input. Hence masked language modelling -- corrupt 15% of positions and
predict those. The consequence for the forward pass is that there is no
causal mask: the anchor checks the attention matrix is NOT triangular,
because an encoder that silently applies a causal mask still runs, still
trains, and is a different model.

**Padding must be masked, and this is where implementations quietly
leak.** A padded position contributes to every other position's softmax
unless it is excluded, so a batch's results depend on how much padding
its longest member happened to need. The mask is applied *before* the
softmax by driving those logits to a large negative value; the anchor
checks that a sequence's output is unchanged when padding is appended.

**Post-norm, as published.** :math:`\mathrm{LayerNorm}(x +
\mathrm{Sublayer}(x))` -- the normalisation comes after the residual
add. The now-common pre-norm variant is a different architecture with
different training dynamics; it is available as an option and is not
the default, because this module is the published one.

References
----------
Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019) "BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding", *Proceedings of NAACL-HLT 2019*, 4171-4186,
doi:10.18653/v1/N19-1423, arXiv:1810.04805. Sec. 3, the architecture
and the masked-LM objective.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
A. N., Kaiser, L. & Polosukhin, I. (2017) "Attention Is All You Need",
*Advances in Neural Information Processing Systems* 30,
arXiv:1706.03762. The encoder block BERT stacks.

Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016) "Layer Normalization",
arXiv:1607.06450.

Hendrycks, D. & Gimpel, K. (2016) "Gaussian Error Linear Units
(GELUs)", arXiv:1606.08415. The activation BERT uses in place of ReLU.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["layer_norm", "multi_head_attention", "encoder_block",
           "bert_encoder", "attention_weights"]

_EPS = 1e-12
_NEG = -1e9


def layer_norm(x, gain=None, bias=None, eps=1e-12):
    """Centre AND scale -- unlike RMSNorm, the mean is subtracted."""
    d = len(x)
    if d == 0:
        raise ValueError("berte: empty vector")
    mu = sum(x) / d
    var = sum((v - mu) ** 2 for v in x) / d
    inv = 1.0 / math.sqrt(var + eps)
    out = [(v - mu) * inv for v in x]
    if gain is not None:
        if len(gain) != d:
            raise ValueError("berte: gain has %d entries for %d "
                             "channels" % (len(gain), d))
        out = [out[i] * gain[i] for i in range(d)]
    if bias is not None:
        out = [out[i] + bias[i] for i in range(d)]
    return out


def attention_weights(Q, K, n_heads, pad_mask=None, causal=False):
    r"""The per-head softmax weights, returned so they can be inspected.

    ``pad_mask[j]`` False means position j is padding and must not be
    attended to by anything -- applied BEFORE the softmax, or the
    normaliser silently includes it.
    """
    L = len(Q)
    d = len(Q[0])
    if d % n_heads != 0:
        raise ValueError("berte: dimension %d is not divisible by %d "
                         "heads" % (d, n_heads))
    hd = d // n_heads
    scale = 1.0 / math.sqrt(hd)
    heads = []
    for h in range(n_heads):
        rows = []
        for i in range(L):
            sc = []
            for j in range(L):
                v = scale * sum(Q[i][h * hd + c] * K[j][h * hd + c]
                                for c in range(hd))
                if pad_mask is not None and not pad_mask[j]:
                    v = _NEG
                if causal and j > i:
                    v = _NEG
                sc.append(v)
            mx = max(sc)
            e = [math.exp(v - mx) for v in sc]
            tot = sum(e)
            rows.append([v / tot for v in e])
        heads.append(rows)
    return heads


def multi_head_attention(Q, K, V, n_heads, pad_mask=None, causal=False):
    """Attention output, concatenated across heads."""
    L = len(Q)
    d = len(Q[0])
    hd = d // n_heads
    w = attention_weights(Q, K, n_heads, pad_mask=pad_mask,
                          causal=causal)
    out = [[0.0] * d for _ in range(L)]
    for h in range(n_heads):
        for i in range(L):
            for c in range(hd):
                out[i][h * hd + c] = sum(
                    w[h][i][j] * V[j][h * hd + c] for j in range(L))
    return out, w


def encoder_block(X, Wq, Wk, Wv, Wo, W1, b1, W2, b2, n_heads,
                  pad_mask=None, gain1=None, bias1=None, gain2=None,
                  bias2=None, pre_norm=False):
    r"""One block: attention, residual, LayerNorm, FFN, residual,
    LayerNorm -- post-norm as published."""
    Xm = k.mat(X)
    L, d = len(Xm), len(Xm[0])

    def proj(row, Wm, b=None):
        v = [sum(row[i] * Wm[i][j] for i in range(len(row)))
             for j in range(len(Wm[0]))]
        return v if b is None else [v[j] + b[j] for j in range(len(v))]

    src = ([layer_norm(Xm[t], gain1, bias1) for t in range(L)]
           if pre_norm else Xm)
    Q = [proj(src[t], Wq) for t in range(L)]
    K = [proj(src[t], Wk) for t in range(L)]
    V = [proj(src[t], Wv) for t in range(L)]
    a, w = multi_head_attention(Q, K, V, n_heads, pad_mask=pad_mask)
    a = [proj(a[t], Wo) for t in range(L)]
    x1 = [[Xm[t][c] + a[t][c] for c in range(d)] for t in range(L)]
    if not pre_norm:
        x1 = [layer_norm(x1[t], gain1, bias1) for t in range(L)]
    src2 = ([layer_norm(x1[t], gain2, bias2) for t in range(L)]
            if pre_norm else x1)
    f = [proj(src2[t], W1, b1) for t in range(L)]
    f = [[k.gelu(v) for v in row] for row in f]
    f = [proj(f[t], W2, b2) for t in range(L)]
    x2 = [[x1[t][c] + f[t][c] for c in range(d)] for t in range(L)]
    if not pre_norm:
        x2 = [layer_norm(x2[t], gain2, bias2) for t in range(L)]
    return x2, w


def bert_encoder(X, blocks, n_heads, pad_mask=None, pre_norm=False):
    """Stack the blocks and return the final states plus every
    block's attention."""
    cur = k.mat(X)
    attn = []
    for b in blocks:
        cur, w = encoder_block(cur, *b, n_heads=n_heads,
                               pad_mask=pad_mask, pre_norm=pre_norm)
        attn.append(w)
    L = len(cur)
    return RichResult(payload={
        "estimate": cur, "output": cur, "attention": attn,
        "pooled": cur[0], "L": L, "d": len(cur[0]),
        "n_layers": len(blocks), "n_heads": n_heads,
        "pre_norm": bool(pre_norm), "bidirectional": True,
        "method": "BERT encoder forward pass, Devlin et al. (2019)",
    })


def cheatsheet():
    return ("berte: encoder block = LayerNorm(x + MHA(x)) then "
            "LayerNorm(x + FFN(x)), post-norm as published, GELU in the "
            "FFN. NO causal mask -- it is bidirectional, which is why "
            "the objective is masked-LM. Padding must be driven to -1e9 "
            "BEFORE the softmax or padded positions leak into every "
            "normaliser and the answer depends on batch shape.")


# compact alias per ledger/NAMING.md
bertencoder = bert_encoder
