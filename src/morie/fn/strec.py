# morie.fn -- function file (rootcoder007/morie)
r"""STAMP: give the last click priority over the session average.

A session-based recommender has no user profile, only the clicks so
far. Earlier neural models summarised that prefix and predicted from
the summary, without allowing for the fact that **interests drift** --
often because of unintended clicks. The paper's example is concrete: a
user who has just clicked a digital camera is likely to act on *that*,
not on whatever dominated the session ten clicks ago.

**Two memories, and the short one gets priority.** The session prefix
is an external memory; the average of its embeddings,

.. math:: m_s = \frac{1}{t}\sum_{i=1}^{t} x_i,

is the user's **general** interest, and the **last click**
:math:`m_t = x_t` is the current one. Both pass through their own
single-layer MLP (identical structure, independent parameters) to give
:math:`h_s` and :math:`h_t`.

**Scoring is trilinear, not a dot product.** With
:math:`\langle a,b,c\rangle = \sum_i a_ib_ic_i = a^\top(b\odot c)`,

.. math:: \hat z_i = \sigma(\langle h_s, h_t, x_i\rangle),

so a candidate must agree with the general interest **and** the
current one at once. A model that added the two representations would
let a strong long-term signal carry a candidate the last click
contradicts; the Hadamard product cannot.

**The attention exists because the average is wrong.** :math:`m_s`
weights every click in the prefix equally, which is exactly what fails
when an unintended click sits in a long session. STAMP replaces it
with an attention-weighted sum,

.. math:: \alpha_i = W_0\,\sigma(W_1 x_i + W_2 x_t + W_3 m_s + b_a),
          \qquad m_a = \sum_{i=1}^{t}\alpha_i x_i,

where each weight sees the item, **the last click**, and the session
average. Note what is absent: no softmax is applied to
:math:`\alpha`, and the weights are not constrained to sum to one.

References
----------
Liu, Q., Zeng, Y., Mokhosi, R. & Zhang, H. (2018) "STAMP: Short-Term
Attention/Memory Priority Model for Session-based Recommendation",
*Proceedings of the 24th ACM SIGKDD International Conference on
Knowledge Discovery & Data Mining (KDD '18)*, 1831-1839,
doi:10.1145/3219819.3219950. [PDF supplied by Vee.] The abstract and
Sec. 1 (that predicting from a session prefix without allowing for
users' interests drifting with time is problematic, the digital-camera
example, and the proposal of a short-term attention/memory priority
model capturing general interests from the long-term memory of the
session context while taking the current interests from the short-term
memory of the last click); Sec. 3.1 (the trilinear product
<a,b,c> = sum a_i b_i c_i = a^T (b (*) c)); Sec. 3.2 (the STMP model
with m_s the average of the external memory, m_t = x_t the last click,
two identically structured MLP cells with independent parameters, the
score z_i = sigma(<h_s, h_t, x_i>), the softmax over candidates and
the cross-entropy loss); and Sec. 3.3 (that treating each item in the
prefix as equally important is problematic for interest drift in long
sessions, and the attention net alpha_i = W_0 sigma(W_1 x_i + W_2 x_t
+ W_3 m_s + b_a) with m_a the attention-weighted sum replacing m_s).

Li, J., Ren, P., Chen, Z., Ren, Z., Lian, T. & Ma, J. (2017) "Neural
Attentive Session-based Recommendation", *CIKM 2017*, 1419-1428,
arXiv:1711.04725. NARM, which the paper distinguishes itself from --
it combines main purpose and sequential behaviour as equally
important, where STAMP explicitly privileges the last click;
implemented in :mod:`narm`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["trilinear", "session_average", "mlp_cell",
           "attention_weights", "stamp_scores"]

_EPS = 1e-12


def trilinear(a, b, c):
    r""":math:`\langle a,b,c\rangle = a^\top(b\odot c)`.

    A candidate has to agree with BOTH representations at once; a sum
    would let one of them carry it alone.
    """
    A = [float(v) for v in k.vec(a)]
    B = [float(v) for v in k.vec(b)]
    C = [float(v) for v in k.vec(c)]
    if not (len(A) == len(B) == len(C)):
        raise ValueError("strec: the three vectors differ in length "
                         "(%d, %d, %d)" % (len(A), len(B), len(C)))
    return sum(A[i] * B[i] * C[i] for i in range(len(A)))


def session_average(embeddings):
    r""":math:`m_s = \frac1t\sum_i x_i` -- the general interest.

    Every click weighted equally, which is precisely what an
    unintended click in a long session breaks.
    """
    X = [[float(v) for v in r] for r in k.mat(embeddings)]
    t = len(X)
    if t < 1:
        raise ValueError("strec: the session prefix is empty")
    d = len(X[0])
    return {"m_s": [sum(X[i][a] for i in range(t)) / t
                    for a in range(d)],
            "m_t": list(X[-1]), "length": t,
            "note": "m_t is the LAST CLICK, and it is also part of "
                    "the external memory"}


def mlp_cell(m, W, b=None, activation="tanh"):
    r"""A single layer: :math:`h = f(Wm + b)`.

    The two cells have identical structure and independent
    parameters.
    """
    v = [float(x) for x in k.vec(m)]
    if len(W[0]) != len(v):
        raise ValueError("strec: the cell expects %d inputs but got "
                         "%d" % (len(W[0]), len(v)))
    bb = [0.0] * len(W) if b is None else [float(x) for x in k.vec(b)]
    z = [bb[o] + sum(W[o][j] * v[j] for j in range(len(v)))
         for o in range(len(W))]
    if activation == "tanh":
        return [math.tanh(x) for x in z]
    if activation == "identity":
        return z
    raise ValueError("strec: activation must be tanh or identity, "
                     "got %r" % (activation,))


def attention_weights(embeddings, W1, W2, W3, W0, b_a=None):
    r""":math:`\alpha_i = W_0\sigma(W_1x_i + W_2x_t + W_3m_s + b_a)`.

    Each weight sees the item, the LAST CLICK and the session average.
    The weights are NOT softmax-normalised -- the paper's composition
    is a plain weighted sum, so :math:`\sum_i\alpha_i` is free.
    """
    X = [[float(v) for v in r] for r in k.mat(embeddings)]
    t = len(X)
    if t < 1:
        raise ValueError("strec: the session prefix is empty")
    d = len(X[0])
    xt = X[-1]
    ms = session_average(X)["m_s"]
    h = len(W1)
    bb = [0.0] * h if b_a is None else [float(v) for v in k.vec(b_a)]

    def sig(x):
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))

    alphas = []
    for i in range(t):
        inner = []
        for o in range(h):
            s = bb[o]
            s += sum(W1[o][j] * X[i][j] for j in range(d))
            s += sum(W2[o][j] * xt[j] for j in range(d))
            s += sum(W3[o][j] * ms[j] for j in range(d))
            inner.append(sig(s))
        alphas.append(sum(W0[o] * inner[o] for o in range(h)))
    m_a = [sum(alphas[i] * X[i][a] for i in range(t))
           for a in range(d)]
    return {"alpha": alphas, "m_a": m_a, "sum_alpha": sum(alphas),
            "m_s": ms,
            "note": "no softmax: the composition is a weighted sum, "
                    "so the weights need not sum to 1"}


def stamp_scores(embeddings, item_table, Ws, Wt, bs=None, bt=None,
                 attention=None):
    r"""Score every candidate by the trilinear composition.

    ``attention`` is the dict from :func:`attention_weights`; without
    it this is STMP, the plain-average baseline the attention was
    introduced to fix.
    """
    X = [[float(v) for v in r] for r in k.mat(embeddings)]
    V = [[float(v) for v in r] for r in k.mat(item_table)]
    base = session_average(X)
    m_s = base["m_s"] if attention is None else attention["m_a"]
    m_t = base["m_t"]
    h_s = mlp_cell(m_s, Ws, bs)
    h_t = mlp_cell(m_t, Wt, bt)

    def sig(x):
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))

    z = [sig(trilinear(h_s, h_t, v)) for v in V]
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    tot = sum(e)
    y = [v / tot for v in e]
    order = sorted(range(len(y)), key=lambda i: -y[i])
    return RichResult(payload={
        "estimate": order[0], "ranking": order, "probability": y,
        "score": z, "h_s": h_s, "h_t": h_t,
        "attention_used": attention is not None,
        "model": "STAMP" if attention is not None else "STMP",
        "method": "short-term attention/memory priority; Liu, Zeng, "
                  "Mokhosi & Zhang (2018)",
        "note": "trilinear, so a candidate must match the general AND "
                "the current interest -- a sum would let one carry it",
    })


def cross_entropy(probability, target_index):
    r"""The training loss: :math:`-\sum_i y_i\log\hat y_i + (1-y_i)
    \log(1-\hat y_i)` with a one-hot target."""
    p = [float(v) for v in k.vec(probability)]
    j = int(target_index)
    if j < 0 or j >= len(p):
        raise ValueError("strec: the target is outside the item "
                         "dictionary")
    tot = 0.0
    for i in range(len(p)):
        yi = 1.0 if i == j else 0.0
        tot += (yi * math.log(max(p[i], _EPS))
                + (1.0 - yi) * math.log(max(1.0 - p[i], _EPS)))
    return -tot


def cheatsheet():
    return ("strec: a session recommender has no profile, only the "
            "clicks -- and interests DRIFT, often from unintended "
            "clicks. Keep TWO memories: m_s, the average of the "
            "session prefix (general interest), and m_t = x_t, the "
            "LAST CLICK (current interest), each through its own MLP "
            "cell. Score TRILINEARLY, sigma(<h_s, h_t, x_i>), so a "
            "candidate must match both at once -- a sum would let a "
            "stale long-term signal override the last click. The "
            "average weights every click equally, which is what breaks "
            "in a long session, so STAMP replaces it with attention "
            "alpha_i = W0 sigma(W1 x_i + W2 x_t + W3 m_s + b_a). No "
            "softmax on alpha.")


# compact alias per ledger/NAMING.md
stamp = stamp_scores
