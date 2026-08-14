# morie.fn -- function file (rootcoder007/morie)
r"""SVD++: which items a user rated is itself a signal.

A latent factor model predicts :math:`\hat r_{ui} = b_{ui} +
q_i^\top p_u` and learns :math:`p_u` only from the ratings the user
gave. But a rating dataset carries a second, weaker signal that costs
nothing to collect: **which** items the user chose to rate at all,
regardless of the score. A user who rated forty horror films has told
you something even if they disliked every one.

**The model adds that signal to the user factor**, not as a separate
term:

.. math:: \hat r_{ui} = b_{ui} + q_i^\top\Big(p_u
          + |N(u)|^{-1/2}\sum_{j\in N(u)} y_j\Big),

with :math:`N(u)` the set of items :math:`u` rated and :math:`y_j` a
second item factor. Adding it *inside* the inner product is what makes
it a modification of the user's taste vector rather than a bias.

**The :math:`|N(u)|^{-1/2}` normalisation is load-bearing.** Without
it a heavy rater's implicit term grows without bound and swamps
:math:`p_u`; with :math:`|N(u)|^{-1}` it would shrink to an average
and stop distinguishing a user with 5 ratings from one with 500. The
square root sits between, and ``implicit_term`` exposes the exponent
so the two failure modes can be seen rather than described.

**Baselines first.** :math:`b_{ui} = \mu + b_u + b_i` absorbs the fact
that some users rate high and some items are widely liked; without it
the factors waste capacity re-learning it.

References
----------
Koren, Y. (2008) "Factorization Meets the Neighborhood: a
Multifaceted Collaborative Filtering Model", *Proceedings of the 14th
ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining (KDD '08)*, 426-434, doi:10.1145/1401890.1401944. [PDF
supplied by Vee.] Sec. 1 and 4: that implicit feedback -- purchase
history, browsing history, search patterns -- indirectly reflects
opinion, and that within a rating dataset a less obvious kind of
implicit data exists, namely WHICH items a user rated regardless of
the rating value, which the paper finds significantly improves
prediction accuracy; and eq. (15), r_ui = b_ui + q_i^T (p_u +
|N(u)|^{-1/2} sum_{j in N(u)} y_j).

Koren, Y., Bell, R. & Volinsky, C. (2009) "Matrix Factorization
Techniques for Recommender Systems", *Computer* 42(8), 30-37,
doi:10.1109/MC.2009.263. The baseline decomposition mu + b_u + b_i.

Hu, Y., Koren, Y. & Volinsky, C. (2008) "Collaborative Filtering for
Implicit Feedback Datasets", *ICDM 2008*, 263-272,
doi:10.1109/ICDM.2008.22. The purely implicit alternative.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["baseline", "implicit_term", "predict", "sgd_step",
           "fit_svdpp"]

_EPS = 1e-12


def baseline(mu, b_user, b_item):
    r""":math:`b_{ui} = \mu + b_u + b_i`."""
    return float(mu) + float(b_user) + float(b_item)


def implicit_term(rated_items, y, exponent=-0.5):
    r""":math:`|N(u)|^{-1/2}\sum_{j\in N(u)} y_j`.

    ``exponent`` is exposed because it is the whole argument: at 0 a
    heavy rater's term swamps :math:`p_u`, at :math:`-1` it becomes a
    mean and forgets how much evidence there was.
    """
    N = list(rated_items)
    if not N:
        return {"term": [0.0] * len(next(iter(y.values()), [0.0])),
                "n_rated": 0,
                "note": "a user with no ratings gets no implicit "
                        "signal"}
    d = len(y[N[0]])
    s = [sum(float(y[j][a]) for j in N) for a in range(d)]
    scale = float(len(N)) ** float(exponent)
    return {"term": [scale * v for v in s], "n_rated": len(N),
            "scale": scale, "exponent": float(exponent),
            "raw_sum": s}


def predict(mu, b_user, b_item, p_u, q_i, rated_items=None, y=None,
            exponent=-0.5):
    r"""The prediction, with the implicit term INSIDE the inner
    product."""
    p = [float(v) for v in k.vec(p_u)]
    q = [float(v) for v in k.vec(q_i)]
    if len(p) != len(q):
        raise ValueError("svdpp: the user and item factors differ in "
                         "width (%d, %d)" % (len(p), len(q)))
    imp = [0.0] * len(p)
    n_rated = 0
    if rated_items and y:
        r = implicit_term(rated_items, y, exponent)
        # width is taken from the factor vector, not from whatever
        # the term happened to return: with no implicit data at all
        # there is no y to read a width from.
        if len(r["term"]) == len(p):
            imp = r["term"]
        n_rated = r["n_rated"]
    eff = [p[a] + imp[a] for a in range(len(p))]
    return {"prediction": baseline(mu, b_user, b_item)
            + sum(q[a] * eff[a] for a in range(len(q))),
            "effective_user_factor": eff, "implicit": imp,
            "n_rated": n_rated,
            "note": "inside the inner product, so it modifies the "
                    "user's TASTE rather than adding a bias"}


def sgd_step(rating, mu, b_user, b_item, p_u, q_i, rated_items, y,
             lr=0.007, reg=0.015, exponent=-0.5):
    r"""One gradient step on the regularised squared error.

    The :math:`y_j` update carries the same :math:`|N(u)|^{-1/2}`
    factor as the forward pass; dropping it there is a silent
    asymmetry that still trains.
    """
    pr = predict(mu, b_user, b_item, p_u, q_i, rated_items, y,
                 exponent)
    e = float(rating) - pr["prediction"]
    a_, r_ = float(lr), float(reg)
    p = [float(v) for v in k.vec(p_u)]
    q = [float(v) for v in k.vec(q_i)]
    d = len(p)
    nb_u = float(b_user) + a_ * (e - r_ * float(b_user))
    nb_i = float(b_item) + a_ * (e - r_ * float(b_item))
    nq = [q[t] + a_ * (e * pr["effective_user_factor"][t]
                       - r_ * q[t]) for t in range(d)]
    npu = [p[t] + a_ * (e * q[t] - r_ * p[t]) for t in range(d)]
    ny = {j: list(y[j]) for j in rated_items}
    scale = float(max(len(list(rated_items)), 1)) ** float(exponent)
    for j in rated_items:
        ny[j] = [y[j][t] + a_ * (e * scale * q[t] - r_ * y[j][t])
                 for t in range(d)]
    return {"error": e, "b_user": nb_u, "b_item": nb_i,
            "p_u": npu, "q_i": nq, "y": ny,
            "note": "the y update carries the same |N(u)|^-1/2 the "
                    "forward pass uses"}


def fit_svdpp(ratings, n_users, n_items, factors=4, epochs=30,
              lr=0.007, reg=0.015, exponent=-0.5, seed=0,
              implicit=True):
    r"""Fit by SGD. ``implicit=False`` gives plain SVD, for
    comparison."""
    R = [(int(u), int(i), float(r)) for u, i, r in ratings]
    if not R:
        raise ValueError("svdpp: no ratings given")
    nu, ni, d = int(n_users), int(n_items), int(factors)
    mu = sum(r for _, _, r in R) / len(R)
    rng = np.random.default_rng(seed)

    def small():
        return [(float(rng.uniform()) - 0.5) * 0.1 for _ in range(d)]

    bu = [0.0] * nu
    bi = [0.0] * ni
    P = [small() for _ in range(nu)]
    Q = [small() for _ in range(ni)]
    Y = {i: small() for i in range(ni)}
    N = {}
    for u, i, _ in R:
        N.setdefault(u, []).append(i)
    hist = []
    for _ in range(int(epochs)):
        se = 0.0
        for (u, i, r) in R:
            items = N[u] if implicit else None
            st = sgd_step(r, mu, bu[u], bi[i], P[u], Q[i],
                          items if items is not None else [],
                          Y if implicit else {}, lr, reg, exponent)
            se += st["error"] ** 2
            bu[u], bi[i] = st["b_user"], st["b_item"]
            P[u], Q[i] = st["p_u"], st["q_i"]
            if implicit:
                for j in st["y"]:
                    Y[j] = st["y"][j]
        hist.append(math.sqrt(se / len(R)))
    return RichResult(payload={
        "estimate": hist[-1], "rmse": hist[-1], "rmse_history": hist,
        "mu": mu, "b_user": bu, "b_item": bi, "P": P, "Q": Q,
        "Y": Y if implicit else None, "implicit": bool(implicit),
        "method": "SVD++; Koren (2008) eq. (15)",
        "note": "which items were rated is a signal even when the "
                "ratings themselves are not used",
    })


def cheatsheet():
    return ("svdpp: a rating dataset carries a SECOND signal for free "
            "-- WHICH items a user rated, regardless of the score. Add "
            "it to the user factor INSIDE the inner product: r_ui = "
            "b_ui + q_i'(p_u + |N(u)|^-1/2 sum_{j in N(u)} y_j), so it "
            "modifies taste rather than adding a bias. The "
            "|N(u)|^-1/2 is load-bearing: at exponent 0 a heavy "
            "rater's term swamps p_u, at -1 it becomes a mean and "
            "forgets how much evidence there was. Baselines mu + b_u + "
            "b_i come first, or the factors waste capacity relearning "
            "them.")


# compact alias per ledger/NAMING.md
svd_plus_plus = fit_svdpp
