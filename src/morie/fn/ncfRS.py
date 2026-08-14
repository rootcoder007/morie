# morie.fn -- function file (rootcoder007/morie)
r"""Neural collaborative filtering: the inner product is a choice.

Matrix factorisation scores a pair by :math:`p_u^\top q_i`. NCF's
argument is that this fixed, un-weighted combination of latent
dimensions is a *modelling assumption* rather than a necessity, and it
limits what user-item similarity structure the model can express.

**GMF: matrix factorisation is a special case, and recovering it
matters.** Take the element-wise product and pass it through a learned
output layer,

.. math:: \hat y_{ui} = a_{out}\big(h^\top (p_u \odot q_i)\big).

With :math:`a_{out}` the identity and :math:`h` the all-ones vector,
this *is* matrix factorisation. Letting :math:`h` be learned makes the
dimensions differently weighted; making :math:`a_{out}` a sigmoid
makes it non-linear. The generalisation is exact, and the anchor
checks the recovery numerically rather than asserting it.

**MLP: concatenation plus depth.** Concatenating :math:`p_u` and
:math:`q_i` and stacking hidden layers lets the model learn the
interaction instead of fixing it -- but concatenation alone accounts
for no interaction at all, which is precisely why the hidden layers
are needed.

**NeuMF: separate embeddings, fused late.** The two pathways are given
*different* embeddings and combined only in the last layer,

.. math:: \hat y_{ui} = \sigma\big(h^\top [\,\phi^{GMF};
          \phi^{MLP}\,]\big),

because forcing them to share one embedding would constrain both to
the same dimension and tie the two models' capacities together.

**The loss follows from the data being implicit.** Interactions are
binary -- observed or not -- so the model is trained with the log
loss under a Bernoulli likelihood, with negatives sampled from the
unobserved entries.

References
----------
He, X., Liao, L., Zhang, H., Nie, L., Hu, X. & Chua, T.-S. (2017)
"Neural Collaborative Filtering", *Proceedings of the 26th
International Conference on World Wide Web (WWW '17)*, 173-182,
doi:10.1145/3038912.3052569. Sec. 3.1 (the general NCF framework and
the log loss with sampled negatives for implicit data), Sec. 3.2
(GMF: eq. (9), and the demonstration that MF is recovered when a_out
is the identity and h is uniform), Sec. 3.3 (MLP over the
concatenation), and Sec. 3.4 (NeuMF: separate GMF and MLP embeddings
fused in the last layer, and why sharing one embedding would limit
the fused model).

Koren, Y., Bell, R. & Volinsky, C. (2009) "Matrix Factorization
Techniques for Recommender Systems", *Computer* 42(8), 30-37,
doi:10.1109/MC.2009.263. The inner-product model being generalised.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["gmf", "mlp_layers", "neumf", "log_loss", "fit_gmf"]

_EPS = 1e-12


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def gmf(p_u, q_i, h=None, activation="sigmoid"):
    r"""Eq. (9): :math:`a_{out}(h^\top(p_u \odot q_i))`.

    ``h=None`` and ``activation="identity"`` recover plain matrix
    factorisation exactly.
    """
    p = [float(v) for v in k.vec(p_u)]
    q = [float(v) for v in k.vec(q_i)]
    if len(p) != len(q):
        raise ValueError("ncfRS: embeddings differ in length (%d, %d)"
                         % (len(p), len(q)))
    hh = [1.0] * len(p) if h is None else [float(v) for v in k.vec(h)]
    z = sum(hh[f] * p[f] * q[f] for f in range(len(p)))
    if activation == "identity":
        return z
    if activation == "sigmoid":
        return _sig(z)
    raise ValueError("ncfRS: activation must be identity or sigmoid, "
                     "got %r" % (activation,))


def mlp_layers(p_u, q_i, Ws, bs):
    r"""Concatenate, then stack ReLU layers.

    Concatenation alone models no interaction; the depth is what
    supplies it.
    """
    z = [float(v) for v in k.vec(p_u)] + \
        [float(v) for v in k.vec(q_i)]
    for l in range(len(Ws)):
        W, b = Ws[l], bs[l]
        z = [max(0.0, b[o] + sum(W[o][i] * z[i]
                                 for i in range(len(z))))
             for o in range(len(b))]
    return z


def neumf(p_gmf, q_gmf, p_mlp, q_mlp, Ws, bs, h):
    r"""Fuse the two pathways in the last layer only."""
    g = [float(a) * float(b) for a, b in
         zip(k.vec(p_gmf), k.vec(q_gmf))]
    m = mlp_layers(p_mlp, q_mlp, Ws, bs)
    cat = list(g) + list(m)
    hh = [float(v) for v in k.vec(h)]
    if len(hh) != len(cat):
        raise ValueError("ncfRS: h has %d entries for a fused vector "
                         "of %d" % (len(hh), len(cat)))
    return {"score": _sig(sum(hh[i] * cat[i]
                              for i in range(len(cat)))),
            "gmf_part": g, "mlp_part": m,
            "note": "separate embeddings per pathway -- sharing one "
                    "would tie both models to the same dimension"}


def log_loss(y, y_hat):
    r"""Bernoulli log loss for binary implicit interactions."""
    p = min(max(float(y_hat), _EPS), 1.0 - _EPS)
    yv = float(y)
    return -(yv * math.log(p) + (1.0 - yv) * math.log(1.0 - p))


def fit_gmf(pos, n_users, n_items, k_dim=8, alpha=0.05, iters=2000,
            n_neg=4, seed=0, learn_h=True):
    r"""GMF by SGD with sampled negatives.

    ``learn_h=False`` freezes :math:`h` at ones, which is matrix
    factorisation with a sigmoid output.
    """
    U, I, K = int(n_users), int(n_items), int(k_dim)
    if U < 1 or I < 2 or K < 1:
        raise ValueError("ncfRS: need at least 1 user, 2 items, 1 "
                         "factor")
    users = sorted(pos)
    if not users:
        raise ValueError("ncfRS: no observed interactions")
    rng = np.random.default_rng(seed)
    P = [[(float(rng.uniform()) - 0.5) * 0.2 for _ in range(K)]
         for _ in range(U)]
    Q = [[(float(rng.uniform()) - 0.5) * 0.2 for _ in range(K)]
         for _ in range(I)]
    h = [1.0] * K
    a = float(alpha)
    hist = []
    for it in range(int(iters)):
        u = users[int(float(rng.uniform()) * len(users)) % len(users)]
        seen = set(pos[u])
        pool = [(i, 1.0) for i in pos[u]]
        for _ in range(int(n_neg)):
            j = int(float(rng.uniform()) * I) % I
            if j not in seen:
                pool.append((j, 0.0))
        for i, y in pool:
            z = sum(h[f] * P[u][f] * Q[i][f] for f in range(K))
            e = _sig(z) - y
            for f in range(K):
                gp = e * h[f] * Q[i][f]
                gq = e * h[f] * P[u][f]
                gh = e * P[u][f] * Q[i][f]
                P[u][f] -= a * gp
                Q[i][f] -= a * gq
                if learn_h:
                    h[f] -= a * gh
        if (it + 1) % max(1, int(iters) // 20) == 0:
            L, n = 0.0, 0
            for uu in users:
                for i in range(I):
                    y = 1.0 if i in set(pos[uu]) else 0.0
                    L += log_loss(y, gmf(P[uu], Q[i], h))
                    n += 1
            hist.append(L / n)
    return RichResult(payload={
        "estimate": (P, Q, h), "P": P, "Q": Q, "h": h,
        "loss_history": hist, "final_loss": hist[-1] if hist else
        float("nan"), "k": K, "learned_h": bool(learn_h),
        "method": "GMF by SGD with sampled negatives; He et al. "
                  "(2017) eq. (9)",
    })


def cheatsheet():
    return ("ncfRS: the inner product is an ASSUMPTION, not a "
            "necessity. GMF = a_out(h' (p_u * q_i)) elementwise, which "
            "IS matrix factorisation when a_out is the identity and h "
            "is all ones -- learning h weights the dimensions, a "
            "sigmoid makes it non-linear. MLP concatenates, and "
            "concatenation alone models NO interaction, which is why "
            "the depth is required. NeuMF gives each pathway its OWN "
            "embedding and fuses only at the last layer. Implicit "
            "data, so log loss with sampled negatives.")


# compact alias per ledger/NAMING.md
neuralcollaborativefiltering = fit_gmf
