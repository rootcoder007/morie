# morie.fn -- function file (rootcoder007/morie)
r"""Factorization machines: interactions under sparsity.

An SVM with a polynomial kernel models pairwise interactions with an
independent parameter :math:`w_{i,j}` per pair. Under the sparsity of
a recommender problem that fails outright: a pair is only estimable
when both features are non-zero in the *same* observation, and almost
no pair ever is. FMs replace the free parameter by a factorised one,

.. math:: \hat y(x) = w_0 + \sum_{i=1}^{n} w_i x_i
          + \sum_{i=1}^{n}\sum_{j=i+1}^{n}
            \langle v_i, v_j\rangle\, x_i x_j ,

so information about one interaction now informs related ones -- the
parameters are coupled through the shared vectors instead of being
independent. That coupling is the whole point, and it is why FMs
estimate interactions in exactly the regime where SVMs cannot.

**Linear time, not quadratic.** The double sum looks
:math:`O(kn^2)`; the identity

.. math:: \sum_{i<j}\langle v_i,v_j\rangle x_i x_j
          = \tfrac12 \sum_{f=1}^{k}\Big[
            \Big(\sum_i v_{i,f}x_i\Big)^2
            - \sum_i v_{i,f}^2 x_i^2\Big]

makes it :math:`O(kn)`, and under sparsity :math:`O(k \bar m_D)` in
the non-zero count. Both routes are implemented and the anchor
requires them to agree to machine precision -- that identity is where
an implementation silently goes wrong.

**Why this matters beyond speed.** Because the model equation is
computable directly, the parameters can be optimised in the primal;
no dual formulation, and no support vectors retained for prediction.
A non-linear SVM's prediction depends on part of the training data;
an FM's does not.

**A general model, not a bespoke one.** Feed it the right feature
vector and it subsumes matrix factorisation, SVD++, PITF and FPMC --
those are FMs with a particular input encoding, not separate
algorithms. ``design_mf`` builds the encoding that recovers plain
matrix factorisation, and the anchor confirms the recovery rather
than asserting it.

References
----------
Rendle, S. (2010) "Factorization Machines", *Proceedings of the Tenth
IEEE International Conference on Data Mining (ICDM 2010)*, 995-1000,
doi:10.1109/ICDM.2010.127. Sec. 1 (FMs estimate interactions under
huge sparsity where SVMs fail; the model equation is computable in
linear time so parameters are estimated directly in the primal
without support vectors). Sec. 3 (eq. (1), the factorised interaction
<v_i, v_j>, the argument that factorisation breaks the independence
of interaction parameters, and Lemma 3.1's linear-time
reformulation). Sec. 4 (the relationship to SVMs). Sec. 5 (matrix
factorisation, SVD++, PITF and FPMC as special cases under particular
input encodings).

Rendle, S., Freudenthaler, C., Gantner, Z. & Schmidt-Thieme, L.
(2009) "BPR: Bayesian Personalized Ranking from Implicit Feedback",
*UAI 2009*, 452-461, arXiv:1205.2618. The pairwise ranking objective
an FM may be trained under; implemented in :mod:`bprMF`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["predict", "predict_naive", "design_mf", "fit_fm",
           "gradient"]

_EPS = 1e-12


def predict_naive(x, w0, w, V):
    r"""Eq. (1) as written -- the :math:`O(kn^2)` double sum."""
    xs = [float(v) for v in k.vec(x)]
    n = len(xs)
    s = float(w0) + sum(w[i] * xs[i] for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            s += sum(V[i][f] * V[j][f] for f in range(len(V[0]))) \
                * xs[i] * xs[j]
    return s


def predict(x, w0, w, V):
    r"""The same value in :math:`O(kn)` by Lemma 3.1."""
    xs = [float(v) for v in k.vec(x)]
    n, kk = len(xs), len(V[0])
    s = float(w0) + sum(w[i] * xs[i] for i in range(n))
    for f in range(kk):
        a = sum(V[i][f] * xs[i] for i in range(n))
        b = sum((V[i][f] * xs[i]) ** 2 for i in range(n))
        s += 0.5 * (a * a - b)
    return s


def gradient(x, V, f, i):
    r""":math:`\partial \hat y/\partial v_{i,f}
    = x_i\sum_j v_{j,f}x_j - v_{i,f}x_i^2`."""
    xs = [float(v) for v in k.vec(x)]
    a = sum(V[j][f] * xs[j] for j in range(len(xs)))
    return xs[i] * a - V[i][f] * xs[i] * xs[i]


def design_mf(u, i, n_users, n_items):
    r"""The encoding under which an FM IS matrix factorisation.

    One indicator for the user, one for the item -- the only non-zero
    interaction is then :math:`\langle v_u, v_i\rangle`.
    """
    x = [0.0] * (int(n_users) + int(n_items))
    x[int(u)] = 1.0
    x[int(n_users) + int(i)] = 1.0
    return x


def fit_fm(X, y, k_dim=4, iters=300, alpha=0.02, lam=0.01, seed=0):
    r"""Least-squares FM by stochastic gradient descent."""
    rows = [[float(v) for v in r] for r in k.mat(X)]
    t = [float(v) for v in k.vec(y)]
    if len(rows) != len(t):
        raise ValueError("fmFM: %d rows but %d targets"
                         % (len(rows), len(t)))
    if not rows:
        raise ValueError("fmFM: no data")
    n, kk = len(rows[0]), int(k_dim)
    if kk < 1:
        raise ValueError("fmFM: k must be at least 1")
    rng = np.random.default_rng(seed)
    w0 = 0.0
    w = [0.0] * n
    V = [[(float(rng.uniform()) - 0.5) * 0.1 for _ in range(kk)]
         for _ in range(n)]
    a, lm = float(alpha), float(lam)
    hist = []
    for it in range(int(iters)):
        for r in range(len(rows)):
            e = predict(rows[r], w0, w, V) - t[r]
            w0 -= a * e
            for i in range(n):
                if rows[r][i] != 0.0:
                    w[i] -= a * (e * rows[r][i] + lm * w[i])
                    for f in range(kk):
                        g = gradient(rows[r], V, f, i)
                        V[i][f] -= a * (e * g + lm * V[i][f])
        hist.append(sum((predict(rows[r], w0, w, V) - t[r]) ** 2
                        for r in range(len(rows))) / len(rows))
    return RichResult(payload={
        "estimate": (w0, w, V), "w0": w0, "w": w, "V": V,
        "mse_history": hist, "final_mse": hist[-1],
        "k": kk, "n_features": n,
        "method": "factorization machine, SGD; Rendle (2010) eq. (1) "
                  "with the linear-time reformulation",
    })


def cheatsheet():
    return ("fmFM: y = w0 + sum w_i x_i + sum_{i<j} <v_i,v_j> x_i x_j. "
            "Factorising the interaction parameter COUPLES pairs that "
            "an SVM treats independently, which is why FMs estimate "
            "interactions under sparsity where SVMs fail -- a free "
            "w_ij needs both features non-zero in the same row, and "
            "almost none are. Lemma 3.1 turns the double sum into "
            "O(kn). Because the model equation is direct, parameters "
            "are learned in the PRIMAL with no support vectors. MF, "
            "SVD++, PITF and FPMC are FMs with a particular input "
            "encoding.")


# compact alias per ledger/NAMING.md
factorizationmachine = fit_fm

# public names resolved by fn/_lazy_map.json
factorization_machines = fit_fm
