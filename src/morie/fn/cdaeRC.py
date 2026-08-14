# morie.fn -- function file (rootcoder007/morie)
r"""Collaborative denoising auto-encoder for top-N recommendation.

A denoising auto-encoder is trained to reconstruct :math:`x` from a
*corrupted* :math:`\tilde x`, which forces the hidden layer to find
robust structure instead of learning the identity. CDAE applies that to
a user's binary preference vector and adds one thing: a **user-specific
input node**.

**That node is the whole idea.** The encoder is

.. math:: z_u = h\big(W^\top \tilde y_u + V_u + b\big),

where :math:`W` is shared across users but :math:`V_u` is unique to
user :math:`u`. Without :math:`V_u` this is an ordinary denoising
auto-encoder over item vectors and the user enters only through which
items they consumed; with it, :math:`W_i` and :math:`V_u` become
distributed representations of item and user, and the model spans a
family that includes several latent-factor recommenders. Reconstruction
is :math:`\hat y_{ui} = f(W_i'^\top z_u + b_i')`.

**Corruption is mask-out/drop-out, and the scaling is not optional.**
Each dimension is zeroed with probability :math:`q`, and the surviving
entries are multiplied by :math:`\delta = 1/(1-q)`. That factor is what
keeps the corruption *unbiased*: :math:`E[\tilde x_d] = x_d`. Dropping
it silently shrinks every input by :math:`(1-q)` -- the anchor checks
the expectation, not the code path.

**Why top-N and not rating prediction.** The preference set is binary:
it records whether an item was preferred, not how much. Training on
observed positives only would let the model predict 1 everywhere, so
the observed set is augmented with sampled negatives -- and the
sampling, not the architecture, is what makes the objective meaningful.

**The loss is a choice, and the paper deliberately proposes no new
one.** Square, log, hinge and cross-entropy are all listed and any
objective fitting the point-wise or pair-wise framework may be used.
Note the trap the paper flags: for log and hinge losses the negative
label must be :math:`-1`, not :math:`0`. All four are implemented, and
the anchor pins each against its closed form.

References
----------
Wu, Y., DuBois, C., Zheng, A. X. & Ester, M. (2016) "Collaborative
Denoising Auto-Encoders for Top-N Recommender Systems", *Proceedings
of the Ninth ACM International Conference on Web Search and Data
Mining (WSDM '16)*, 153-162, doi:10.1145/2835776.2835837. Sec. 2 (the
point-wise and pair-wise objective framework, Table 1's placement of
MF/BPR-MF/SLIM/FISM/WRMF within it, the four loss functions and the
warning that log and hinge losses need y = -1 for negatives, and the
need to augment the observed set with sampled negatives to avoid the
trivial all-ones model). Sec. 2.3 (the auto-encoder, tied weights, and
the denoising auto-encoder with mask-out/drop-out corruption scaled by
delta = 1/(1-q) to keep it unbiased). Sec. 3 (CDAE: eqs. (9)-(13), the
user-specific node V_u, and Algorithm 1's SGD with negative sampling).

Vincent, P., Larochelle, H., Bengio, Y. & Manzagol, P.-A. (2008)
"Extracting and composing robust features with denoising
autoencoders", *ICML 2008*, 1096-1103, doi:10.1145/1390156.1390294.
The denoising auto-encoder.

Rendle, S., Freudenthaler, C., Gantner, Z. & Schmidt-Thieme, L.
(2009) "BPR: Bayesian Personalized Ranking from Implicit Feedback",
*UAI 2009*, 452-461, arXiv:1205.2618. The pair-wise objective in
Table 1; implemented in :mod:`bprMF`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["corrupt", "encode", "decode", "loss", "fit_cdae",
           "recommend"]

_EPS = 1e-12
_LOSSES = ("square", "log", "hinge", "cross_entropy")
_ACTS = ("sigmoid", "identity", "tanh")


def _act(name, x):
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-x)) if x >= -700 else 0.0
    if name == "identity":
        return x
    if name == "tanh":
        return math.tanh(x)
    raise ValueError("cdaeRC: activation must be one of %s, got %r"
                     % (", ".join(_ACTS), name))


def _dact(name, y):
    if name == "sigmoid":
        return y * (1.0 - y)
    if name == "identity":
        return 1.0
    return 1.0 - y * y


def corrupt(y, q, rng):
    r"""Eq. (9): zero each entry with probability :math:`q`, scale the
    survivors by :math:`1/(1-q)`.

    The scaling keeps :math:`E[\tilde y] = y`; without it every input
    is shrunk by :math:`(1-q)`.
    """
    qq = float(q)
    if not 0.0 <= qq < 1.0:
        raise ValueError("cdaeRC: q must lie in [0,1), got %r" % (q,))
    d = 1.0 / (1.0 - qq)
    return [0.0 if float(rng.uniform()) < qq else d * float(v)
            for v in y]


def encode(y_tilde, W, V_u, b, activation="sigmoid"):
    r"""Eq. (10): :math:`z_u = h(W^\top \tilde y_u + V_u + b)`."""
    K = len(b)
    z = []
    for f in range(K):
        s = b[f] + V_u[f]
        for i in range(len(y_tilde)):
            if y_tilde[i] != 0.0:
                s += W[i][f] * y_tilde[i]
        z.append(_act(activation, s))
    return z


def decode(z, Wp, bp, items=None, activation="sigmoid"):
    r"""Eq. (11): :math:`\hat y_{ui} = f(W_i'^\top z_u + b_i')`."""
    idx = range(len(bp)) if items is None else list(items)
    return {i: _act(activation,
                    bp[i] + sum(Wp[i][f] * z[f]
                                for f in range(len(z))))
            for i in idx}


def loss(y, y_hat, kind="square"):
    r"""The four losses of Sec. 2.

    For ``log`` and ``hinge`` the negative label must be :math:`-1`;
    passing :math:`0` is rejected rather than silently mis-scored.
    """
    if kind not in _LOSSES:
        raise ValueError("cdaeRC: loss must be one of %s, got %r"
                         % (", ".join(_LOSSES), kind))
    yv, yh = float(y), float(y_hat)
    if kind in ("log", "hinge") and yv == 0.0:
        raise ValueError("cdaeRC: the %s loss needs y = -1 for "
                         "negatives, not 0" % kind)
    if kind == "square":
        return 0.5 * (yv - yh) ** 2
    if kind == "log":
        return math.log(1.0 + math.exp(-yv * yh)) \
            if -yv * yh < 700 else -yv * yh
    if kind == "hinge":
        return max(0.0, 1.0 - yv * yh)
    p = 1.0 / (1.0 + math.exp(-yh)) if yh >= -700 else 0.0
    p = min(max(p, _EPS), 1.0 - _EPS)
    return -yv * math.log(p) - (1.0 - yv) * math.log(1.0 - p)


def fit_cdae(pos, n_users, n_items, k_dim=8, q=0.2, alpha=0.05,
             lam=0.01, iters=30, n_neg=5, seed=0,
             activation="sigmoid", init_scale=0.1):
    r"""Algorithm 1: SGD with mask-out corruption and negative
    sampling.

    The squared :math:`L_2` penalty of eq. (13) is applied to every
    parameter block.
    """
    U, I, K = int(n_users), int(n_items), int(k_dim)
    if U < 1 or I < 2 or K < 1:
        raise ValueError("cdaeRC: need at least 1 user, 2 items and 1 "
                         "hidden node")
    rng = np.random.default_rng(seed)

    def rand():
        return (float(rng.uniform()) - 0.5) * 2.0 * init_scale

    W = [[rand() for _ in range(K)] for _ in range(I)]
    Wp = [[rand() for _ in range(K)] for _ in range(I)]
    V = [[rand() for _ in range(K)] for _ in range(U)]
    b = [0.0] * K
    bp = [0.0] * I
    a, lm = float(alpha), float(lam)
    hist = []
    for it in range(int(iters)):
        tot = 0.0
        for u in range(U):
            seen = sorted(set(int(v) for v in pos.get(u, [])))
            if not seen:
                continue
            y = [1.0 if i in set(seen) else 0.0 for i in range(I)]
            yt = corrupt(y, q, rng)
            z = encode(yt, W, V[u], b, activation)
            neg = []
            guard = 0
            while len(neg) < int(n_neg) and guard < 100 * int(n_neg):
                j = int(float(rng.uniform()) * I) % I
                if j not in set(seen):
                    neg.append(j)
                guard += 1
            tgt = seen + neg
            out = decode(z, Wp, bp, tgt, activation)
            dz = [0.0] * K
            for i in tgt:
                yi = 1.0 if i in set(seen) else 0.0
                e = (out[i] - yi) * _dact(activation, out[i])
                tot += loss(yi, out[i], "square")
                for f in range(K):
                    dz[f] += e * Wp[i][f]
                    Wp[i][f] -= a * (e * z[f] + lm * Wp[i][f])
                bp[i] -= a * e
            dpre = [dz[f] * _dact(activation, z[f]) for f in range(K)]
            for i in range(I):
                if yt[i] != 0.0:
                    for f in range(K):
                        W[i][f] -= a * (dpre[f] * yt[i]
                                        + lm * W[i][f])
            for f in range(K):
                V[u][f] -= a * (dpre[f] + lm * V[u][f])
                b[f] -= a * dpre[f]
        hist.append(tot)
    return RichResult(payload={
        "estimate": (W, Wp, V, b, bp), "W": W, "W_prime": Wp,
        "V": V, "b": b, "b_prime": bp, "loss_history": hist,
        "final_loss": hist[-1] if hist else float("nan"),
        "k": K, "q": float(q), "n_neg": int(n_neg),
        "activation": activation,
        "method": "CDAE; Wu, DuBois, Zheng & Ester (2016) eqs. "
                  "(9)-(13), Algorithm 1",
        "note": "V_u is the user-specific input node -- without it "
                "this is an ordinary denoising auto-encoder over item "
                "vectors",
    })


def recommend(model, pos, u, n_items, top_k=5, activation="sigmoid"):
    r"""Score every unseen item for one user, uncorrupted at test
    time."""
    W, Wp, V, b, bp = model["W"], model["W_prime"], model["V"], \
        model["b"], model["b_prime"]
    seen = set(int(v) for v in pos.get(u, []))
    y = [1.0 if i in seen else 0.0 for i in range(int(n_items))]
    z = encode(y, W, V[int(u)], b, activation)
    out = decode(z, Wp, bp, None, activation)
    s = [(i, out[i]) for i in range(int(n_items)) if i not in seen]
    s.sort(key=lambda t: -t[1])
    return {"ranking": s[:int(top_k)], "n_scored": len(s)}


def cheatsheet():
    return ("cdaeRC: a denoising auto-encoder over a user's BINARY "
            "preference vector, plus a USER-SPECIFIC input node V_u -- "
            "that node is what separates it from a plain DAE and makes "
            "W_i, V_u item and user embeddings. Corruption is "
            "mask-out with probability q, survivors scaled by "
            "1/(1-q) so the corruption is UNBIASED. Positives only "
            "would train the all-ones model, so negatives are SAMPLED. "
            "Four losses offered; log and hinge need the negative "
            "label to be -1, not 0.")


# compact alias per ledger/NAMING.md
collaborativedenoisingautoencoder = fit_cdae
