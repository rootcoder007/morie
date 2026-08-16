"""Log-scaled attention: log-length logits and scaled cosine similarity.

Ordinary attention divides the query-key inner product by the square root
of the head dimension and softmaxes the result. That constant is chosen
so the logits have roughly unit variance at initialisation, and it does
not depend on how many keys there are. Two separate lines of work in 2022
showed that this is the wrong invariance, from opposite directions, and
both fixes are here.

The FIRST is about sequence length. With a fixed logit scale, the softmax
over n keys spreads its mass over n positions, so the weight any one
position can hold decays like 1/n and the model becomes less confident
the longer the input gets -- which is a theorem, not an accident, and it
is why a transformer trained on short strings misclassifies long ones.
Chiang and Cholak's fix is one factor:

    Att(q, K, V) = V' softmax( (log n / sqrt(d)) K q )

Multiplying the logits by log n keeps the attention distribution's
sharpness roughly constant as n grows. This is the "log-scaled" route,
and it is what the module is named for.

The SECOND is about amplitude. In large vision models the learnt
attention maps of some blocks come to be dominated by a few pixel pairs,
because the inner product grows with the norms of the activations and
those norms grow with depth. Swin Transformer V2 replaces the inner
product with a cosine, which is normalised by construction:

    Sim(q_i, k_j) = cos(q_i, k_j) / tau + B_ij

with tau learnable, not shared across heads or layers, and held above
0.01 -- the floor matters, because tau appears in a denominator and a
gradient step that takes it to zero produces infinities rather than a
sharp distribution. The same paper's log-spaced coordinates,

    dx_hat = sign(dx) log(1 + |dx|)

are the other half of the idea: they compress the relative-position range
so a bias learnt on an eight-by-eight window extrapolates to a
sixteen-by-sixteen one. On that window the raw range [-7, 7] becomes
[-2.079, 2.079], which is a number the paper states and this module
checks against.

Both compose, and the composition -- a cosine similarity, log-length
scaled -- is the route to reach for when a model must generalise across
both amplitude and length.

A note on provenance. The ledger row for this module cites "Yu et al
(2022)", which does not resolve to any paper stating this method; the two
that do state it are cited below and the implementation follows them.
Nothing here is taken from the ledger's own description.

References
  Chiang, D. and Cholak, P. (2022) "Overcoming a theoretical limitation
    of self-attention." Proceedings of the 60th Annual Meeting of the
    Association for Computational Linguistics (Volume 1: Long Papers),
    7654-7664, Dublin. arXiv:2202.12172. Their equation (2), the
    log-length scaled attention, and section 5.3 for why the
    unscaled version loses confidence as n grows.
  Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao,
    Y., Zhang, Z., Dong, L., Wei, F. and Guo, B. (2022) "Swin
    Transformer V2: scaling up capacity and resolution." Proceedings of
    the IEEE/CVF Conference on Computer Vision and Pattern Recognition
    (CVPR), 12009-12019. arXiv:2111.09883. Their equation (2), scaled
    cosine attention with the tau > 0.01 floor, and equation (4), the
    log-spaced relative coordinates.
  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
    A.N., Kaiser, L. and Polosukhin, I. (2017) "Attention is all you
    need." Advances in Neural Information Processing Systems 30. The
    1/sqrt(d) baseline both papers are correcting.
  Hahn, M. (2020) "Theoretical limitations of self-attention in neural
    sequence models." Transactions of the ACL 8, 156-171. The lemma that
    the log-length scaling is answering.
"""

import math

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["vit2lf", "vit2_log_attention", "attention_logits", "softmax_rows",
           "log_spaced_coords", "relative_bias", "row_entropy", "MODES",
           "TAU_FLOOR", "cheatsheet"]

MODES = ("dot", "logn", "cosine", "logn_cosine")

# Swin V2 holds the cosine temperature above this. It sits in a
# denominator, so a gradient step that reaches zero gives infinities
# rather than a sharp distribution.
TAU_FLOOR = 0.01


def _norm(v):
    return math.sqrt(_w.dot(v, v))


def attention_logits(q, k, mode="dot", tau=1.0, bias=None, n=None,
                     tau_floor=TAU_FLOOR):
    """The pre-softmax scores, one row per query.

    "dot"          the inner product over the square root of the head
                   dimension.
    "logn"         the same, multiplied by log n. n defaults to the
                   number of KEYS, which is the number of terms the
                   softmax is spreading its mass over.
    "cosine"       the cosine of the angle, over tau. No square root of
                   d appears: a cosine is already normalised, which is
                   the whole point of using one.
    "logn_cosine"  the cosine route with the log-length factor as well.

    An additive bias is applied after the scaling in every route, as both
    papers write it -- a bias scaled along with the logits would change
    meaning with the sequence length, which is exactly what the position
    bias must not do.
    """
    if mode not in MODES:
        raise ValueError("mode must be one of %r" % (MODES,))
    nq = len(q)
    nk = len(k)
    d = len(q[0])
    if any(len(r) != d for r in q) or any(len(r) != d for r in k):
        raise ValueError("queries and keys must share one head dimension")
    nn = float(nk if n is None else n)
    if nn <= 0.0:
        raise ValueError("the length used for the log scaling must be "
                         "positive")
    if mode in ("cosine", "logn_cosine"):
        t = float(tau)
        if t <= 0.0:
            raise ValueError("the cosine temperature must be positive")
        if t < tau_floor:
            t = tau_floor
        base = 1.0 / t
    else:
        base = 1.0 / math.sqrt(float(d))
    scale = base * math.log(nn) if mode in ("logn", "logn_cosine") else base

    qn = [_norm(r) for r in q] if mode in ("cosine", "logn_cosine") else None
    kn = [_norm(r) for r in k] if mode in ("cosine", "logn_cosine") else None
    out = []
    for i in range(nq):
        row = []
        for j in range(nk):
            s = _w.dot(q[i], k[j])
            if qn is not None:
                den = qn[i] * kn[j]
                # A zero vector has no direction, so it has no cosine
                # with anything. Reporting zero is the only answer that
                # does not invent one.
                s = 0.0 if den <= 0.0 else s / den
            s = s * scale
            if bias is not None:
                s = s + float(bias[i][j])
            row.append(s)
        out.append(row)
    return out, scale


def softmax_rows(logits, mask=None, neg=-1e30):
    """Row-wise softmax, max-shifted, with a compensated denominator.

    Masked entries are removed before the shift rather than pushed to a
    large negative number and exponentiated, so a fully masked row is an
    error instead of a silently uniform one.
    """
    out = []
    for i, row in enumerate(logits):
        live = [j for j in range(len(row))
                if mask is None or mask[i][j]]
        if not live:
            raise ValueError("row %d is masked out entirely" % i)
        mx = row[live[0]]
        for j in live:
            if row[j] > mx:
                mx = row[j]
        ex = [0.0] * len(row)
        for j in live:
            ex[j] = math.exp(row[j] - mx)
        tot = _w.csum(ex[j] for j in live)
        out.append([ex[j] / tot for j in range(len(row))])
    return out


def row_entropy(w):
    """Shannon entropy of each attention row, in nats.

    This is the quantity the log-length scaling exists to hold roughly
    constant, so it is worth reporting rather than leaving to be
    recomputed.
    """
    out = []
    for row in w:
        terms = [-p * math.log(p) for p in row if p > 0.0]
        out.append(_w.csum(terms) if terms else 0.0)
    return out


def log_spaced_coords(dx, dy):
    """Swin V2's log-spaced relative coordinates.

    sign(d) log(1 + |d|), which is odd, zero at zero, and compresses the
    range so a bias learnt at one window size extrapolates to another. On
    an eight-by-eight window the raw range [-7, 7] becomes
    [-2.079, 2.079], the figure the paper quotes.
    """
    def f(v):
        v = float(v)
        s = 0.0 if v == 0.0 else (1.0 if v > 0.0 else -1.0)
        return s * math.log1p(abs(v))
    return f(dx), f(dy)


def relative_bias(coords, table, window, log_spaced=True):
    """Look a relative position up in a bias table.

    `coords` gives each position's (x, y); `table` is indexed by the
    relative offset shifted into [0, 2 window - 2]. With `log_spaced` the
    offsets are transformed first and the table is read at the nearest
    tabulated log-spaced offset, which is what the meta-network
    approximates continuously.
    """
    n = len(coords)
    span = 2 * int(window) - 1
    if len(table) != span or any(len(r) != span for r in table):
        raise ValueError("the table must be (2 window - 1) square")
    grid = [log_spaced_coords(t - (window - 1), 0)[0] for t in range(span)]
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            if log_spaced:
                lx, ly = log_spaced_coords(dx, dy)
                a = min(range(span), key=lambda t: (abs(grid[t] - lx), t))
                b = min(range(span), key=lambda t: (abs(grid[t] - ly), t))
            else:
                a = int(dx) + window - 1
                b = int(dy) + window - 1
                if a < 0 or a >= span or b < 0 or b >= span:
                    raise ValueError("a relative offset falls outside the "
                                     "table")
            row.append(float(table[a][b]))
        out.append(row)
    return out


def vit2_log_attention(q, k, v, mode="logn", tau=1.0, bias=None, mask=None,
                       n=None, tau_floor=TAU_FLOOR):
    """Attention with the logits scaled by log n, by a cosine, or by both.

    Parameters
    ----------
    q : sequence of sequences
        Queries, one row per query position.
    k : sequence of sequences
        Keys, one row per key position, sharing the head dimension.
    v : sequence of sequences
        Values, one row per key position.
    mode : str
        A member of MODES.
    tau : float
        The cosine temperature. Held at or above `tau_floor`.
    bias : sequence of sequences or None
        An additive position bias, applied after the scaling.
    mask : sequence of sequences or None
        True where a key is visible to a query.
    n : float or None
        The length used for the log scaling. Defaults to the number of
        keys.
    tau_floor : float
        The lower bound on tau.

    Returns
    -------
    RichResult
        The attention weights, the context vectors, the logits, the
        scale actually applied, the per-row entropy and the largest
        weight -- the last two being how you see the scaling working.

    References
    ----------
    Chiang and Cholak (2022) ACL, 7654-7664, equation (2); Liu et al.
    (2022) CVPR, 12009-12019, equations (2) and (4).
    """
    qq = [[float(x) for x in r] for r in q]
    kk = [[float(x) for x in r] for r in k]
    vv = [[float(x) for x in r] for r in v]
    if not qq or not kk or not vv:
        raise ValueError("queries, keys and values must be non-empty")
    if len(vv) != len(kk):
        raise ValueError("there must be one value per key")
    logits, scale = attention_logits(qq, kk, mode, tau, bias, n, tau_floor)
    w = softmax_rows(logits, mask)
    dv = len(vv[0])
    ctx = []
    for i in range(len(qq)):
        ctx.append([_w.csum(w[i][j] * vv[j][t] for j in range(len(kk)))
                    for t in range(dv)])
    ent = row_entropy(w)
    mx = [max(row) for row in w]
    return RichResult(payload={
        "weights": w,
        "context": ctx,
        "logits": logits,
        "scale": scale,
        "entropy": ent,
        "max_weight": mx,
        "mean_entropy": _w.csum(ent) / len(ent),
        "estimate": _w.csum(mx) / len(mx),
        "se": max(ent) - min(ent),
        "n_query": len(qq),
        "n_key": len(kk),
        "d": len(qq[0]),
        "d_value": dv,
        "tau": max(float(tau), tau_floor) if mode in ("cosine",
                                                      "logn_cosine")
                else float("nan"),
        "mode": mode,
        "method": "log-scaled attention",
    })


vit2lf = vit2_log_attention


def cheatsheet():
    return ("vit2lf: log-scaled attention. modes " + ", ".join(MODES)
            + "; log-length logits (Chiang-Cholak) and scaled cosine "
              "similarity (Swin V2)")
