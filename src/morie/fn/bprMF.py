# morie.fn -- function file (rootcoder007/morie)
r"""BPR: optimising the ranking, not the score.

Implicit feedback is positive-only. The usual move is to label the
observed pairs :math:`(u,i) \in S` as 1, everything else as 0, and fit.
The paper's objection to that is sharp: the elements the model is
*supposed to rank in the future* -- all of :math:`(U \times I)
\setminus S` -- are handed to the learner as negatives. A model with
enough capacity to fit that training data exactly cannot rank at all,
because it predicts 0 everywhere. Such methods only produce rankings
because regularisation stops them fitting.

**So use pairs.** From :math:`S`, assume a user prefers what they
have seen over what they have not:

.. math:: D_S := \{(u,i,j) \mid i \in I_u^+ \wedge
          j \in I \setminus I_u^+\}.

Two items both seen, or both unseen, yield nothing -- and the unseen
pairs are exactly the ones to be ranked at prediction time, so they
are correctly left out of training rather than labelled negative.

**The criterion.** With
:math:`p(i >_u j \mid \Theta) = \sigma(\hat x_{uij})` and a Gaussian
prior :math:`\Theta \sim N(0, \lambda_\Theta I)`,

.. math:: \mathrm{BPR\text{-}Opt} = \sum_{(u,i,j)\in D_S}
          \ln \sigma(\hat x_{uij}) - \lambda_\Theta \|\Theta\|^2 .

**Its relation to AUC is exact and worth stating.** Per-user AUC is
:math:`\sum_{(u,i,j)} z_u\,\delta(\hat x_{uij} > 0)`. Apart from the
normalising constant, BPR-Opt differs *only* in the loss: AUC uses the
non-differentiable Heaviside :math:`\delta(x>0)`, BPR uses
:math:`\ln\sigma(x)`. Replacing the Heaviside with a similarly shaped
function is common practice and usually heuristic; here the
substitution falls out of the maximum-likelihood derivation.

**Learning.** LearnBPR is stochastic gradient ascent over triples
drawn from :math:`D_S` -- drawn, not enumerated, because
:math:`|D_S|` is enormous and sweeping user-wise or item-wise makes
consecutive updates dependent on the same item. The model only has to
supply :math:`\partial \hat x_{uij}/\partial \theta`; for matrix
factorisation with :math:`\hat x_{ui} = \langle w_u, h_i\rangle` and
:math:`\hat x_{uij} = \hat x_{ui} - \hat x_{uj}`, that is
:math:`h_{if} - h_{jf}` for :math:`w_{uf}`, :math:`w_{uf}` for
:math:`h_{if}`, and :math:`-w_{uf}` for :math:`h_{jf}`.

**One printed sign is wrong, and it is not cosmetic.** Figure 4 gives

.. math:: \Theta \leftarrow \Theta + \alpha\Big(
          \frac{e^{-\hat x_{uij}}}{1+e^{-\hat x_{uij}}}
          \cdot \frac{\partial}{\partial\Theta}\hat x_{uij}
          + \lambda_\Theta \cdot \Theta\Big),

but ascending :math:`\sum \ln\sigma - \lambda\|\Theta\|^2` requires
:math:`-\lambda_\Theta\Theta`: as printed, the regulariser *grows* the
parameters at every step. ``learn_bpr`` uses the correct sign and
``regularizer_sign="paper"`` reproduces the printed one, so the
divergence can be seen rather than argued about.

References
----------
Rendle, S., Freudenthaler, C., Gantner, Z. & Schmidt-Thieme, L.
(2009) "BPR: Bayesian Personalized Ranking from Implicit Feedback",
*Proceedings of the Twenty-Fifth Conference on Uncertainty in
Artificial Intelligence (UAI 2009)*, 452-461, arXiv:1205.2618.
Sec. 3 (the objection to labelling all unobserved pairs negative, and
the construction of D_S). Sec. 4.1 (the likelihood
sigma(x_uij), the Gaussian prior, and BPR-Opt). Sec. 4.1.1 (the AUC
analogy: identical but for the Heaviside vs ln sigma loss). Sec. 4.2
and Figure 4 (LearnBPR by bootstrap sampling stochastic gradient
descent). Sec. 4.3 (matrix factorisation, x_uij = x_ui - x_uj, and
the three derivative cases).

Koren, Y., Bell, R. & Volinsky, C. (2009) "Matrix Factorization
Techniques for Recommender Systems", *Computer* 42(8), 30-37,
doi:10.1109/MC.2009.263. The factorisation model class BPR is applied
to here.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sigmoid", "predict", "bpr_opt", "auc", "learn_bpr",
           "recommend"]

_EPS = 1e-12
_SIGNS = ("correct", "paper")


def sigmoid(x):
    r""":math:`\sigma(x) = 1/(1+e^{-x})`, overflow-safe."""
    v = float(x)
    if v >= 0.0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def predict(W, H, u, i):
    r""":math:`\hat x_{ui} = \langle w_u, h_i\rangle`."""
    wu, hi = W[int(u)], H[int(i)]
    return sum(wu[f] * hi[f] for f in range(len(wu)))


def _triples(pos, n_items):
    """:math:`D_S` -- every (seen, unseen) pair, per user."""
    out = []
    for u in sorted(pos):
        seen = set(pos[u])
        for i in seen:
            for j in range(n_items):
                if j not in seen:
                    out.append((u, i, j))
    return out


def bpr_opt(W, H, pos, n_items, lam=0.01):
    r""":math:`\sum \ln\sigma(\hat x_{uij}) - \lambda\|\Theta\|^2`."""
    tot = 0.0
    for u, i, j in _triples(pos, n_items):
        x = predict(W, H, u, i) - predict(W, H, u, j)
        tot += math.log(max(sigmoid(x), _EPS))
    norm = sum(v * v for r in W for v in r) + \
        sum(v * v for r in H for v in r)
    return {"bpr_opt": tot - float(lam) * norm, "loglik": tot,
            "penalty": float(lam) * norm, "n_triples":
                len(_triples(pos, n_items))}


def auc(W, H, pos, n_items):
    r"""Eq. (1): :math:`\sum z_u \delta(\hat x_{uij} > 0)`.

    The indicator is *strict*, so a tie scores zero -- which is what
    makes an all-equal model score 0, not 0.5.
    """
    per, tot = {}, 0.0
    users = sorted(pos)
    if not users:
        raise ValueError("bprMF: no users with positive feedback")
    for u in users:
        seen = set(pos[u])
        neg = [j for j in range(n_items) if j not in seen]
        if not seen or not neg:
            raise ValueError("bprMF: user %r has no comparable pair"
                             % (u,))
        c = 0
        for i in seen:
            for j in neg:
                if predict(W, H, u, i) - predict(W, H, u, j) > 0.0:
                    c += 1
        per[u] = c / float(len(seen) * len(neg))
        tot += per[u]
    return {"auc": tot / len(users), "per_user": per,
            "note": "delta(x > 0) is strict: ties count as wrong"}


def learn_bpr(pos, n_users, n_items, k_dim=8, alpha=0.05, lam=0.01,
              iters=2000, seed=0, regularizer_sign="correct",
              init_scale=0.1):
    r"""LearnBPR: bootstrap-sampled stochastic gradient ascent.

    ``regularizer_sign="paper"`` reproduces the printed Figure 4
    update, whose ``+lambda*Theta`` term diverges; the default is the
    sign that actually ascends BPR-Opt.
    """
    if regularizer_sign not in _SIGNS:
        raise ValueError("bprMF: regularizer_sign must be one of %s, "
                         "got %r" % (", ".join(_SIGNS),
                                     regularizer_sign))
    U, I, K = int(n_users), int(n_items), int(k_dim)
    if U < 1 or I < 2 or K < 1:
        raise ValueError("bprMF: need at least 1 user, 2 items and 1 "
                         "factor")
    users = sorted(pos)
    if not users:
        raise ValueError("bprMF: no positive feedback given")
    rng = np.random.default_rng(seed)
    W = [[(float(rng.uniform()) - 0.5) * 2.0 * init_scale
          for _ in range(K)] for _ in range(U)]
    H = [[(float(rng.uniform()) - 0.5) * 2.0 * init_scale
          for _ in range(K)] for _ in range(I)]
    sgn = -1.0 if regularizer_sign == "correct" else 1.0
    a, lm = float(alpha), float(lam)
    hist = []
    for it in range(int(iters)):
        u = users[int(float(rng.uniform()) * len(users)) % len(users)]
        seen = list(pos[u])
        i = seen[int(float(rng.uniform()) * len(seen)) % len(seen)]
        j = int(float(rng.uniform()) * I) % I
        guard = 0
        while j in set(pos[u]) and guard < 100:
            j = int(float(rng.uniform()) * I) % I
            guard += 1
        if j in set(pos[u]):
            continue
        x = predict(W, H, u, i) - predict(W, H, u, j)
        g = sigmoid(-x)
        for f in range(K):
            wuf, hif, hjf = W[u][f], H[i][f], H[j][f]
            W[u][f] = wuf + a * (g * (hif - hjf) + sgn * lm * wuf)
            H[i][f] = hif + a * (g * wuf + sgn * lm * hif)
            H[j][f] = hjf + a * (g * (-wuf) + sgn * lm * hjf)
        if (it + 1) % max(1, int(iters) // 20) == 0:
            hist.append(bpr_opt(W, H, pos, I, lm)["bpr_opt"])
    norm = math.sqrt(sum(v * v for r in W for v in r)
                     + sum(v * v for r in H for v in r))
    return RichResult(payload={
        "estimate": (W, H), "W": W, "H": H, "k": K,
        "bpr_opt_history": hist,
        "final_bpr_opt": hist[-1] if hist else float("nan"),
        "auc": auc(W, H, pos, I)["auc"], "param_norm": norm,
        "regularizer_sign": regularizer_sign,
        "method": "LearnBPR, bootstrap SGD; Rendle et al. (2009) "
                  "Fig. 4",
        "caveat": ("the printed update adds +lambda*Theta, which "
                   "grows the parameters; this run used that sign"
                   if regularizer_sign == "paper"
                   else "regulariser sign corrected to -lambda*Theta, "
                        "which is what ascending BPR-Opt requires"),
    })


def recommend(W, H, u, n_items, top_k=5, exclude=()):
    r"""Rank items for one user by :math:`\hat x_{ui}`."""
    ex = set(int(v) for v in exclude)
    s = [(i, predict(W, H, u, i)) for i in range(int(n_items))
         if i not in ex]
    s.sort(key=lambda t: -t[1])
    return {"ranking": s[:int(top_k)], "n_scored": len(s)}


def cheatsheet():
    return ("bprMF: implicit feedback is positive-only, and labelling "
            "every unobserved pair NEGATIVE trains the model to "
            "predict 0 on exactly the items it must rank later. Use "
            "TRIPLES instead: D_S = {(u,i,j) : i seen, j unseen}. "
            "BPR-Opt = sum ln sigma(x_ui - x_uj) - lambda||Theta||^2, "
            "which is per-user AUC with the Heaviside replaced by "
            "ln sigma -- and that substitution comes from the MLE, not "
            "from convenience. LearnBPR SAMPLES triples rather than "
            "sweeping them. The printed update's +lambda*Theta is a "
            "sign error and diverges.")


# compact alias per ledger/NAMING.md
bayesianpersonalizedranking = learn_bpr

# public names resolved by fn/_lazy_map.json
bpr_mf = learn_bpr
bprmf = learn_bpr
