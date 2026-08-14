# morie.fn -- function file (rootcoder007/morie)
r"""Implicit feedback: preference and confidence are different things.

Explicit ratings say how much a user liked something. Implicit
feedback -- watch time, purchase counts, clicks -- does not. The
paper's central distinction is that the *numerical value* of implicit
feedback measures **confidence**, not preference: the most loved film
may be watched once, while a merely liked series is watched weekly. A
larger number is not a stronger preference; it is stronger evidence.

So split the observation in two. Preference is binary,

.. math:: p_{ui} = \begin{cases}1 & r_{ui} > 0\\
          0 & r_{ui} = 0\end{cases},

and confidence grows with the evidence,

.. math:: c_{ui} = 1 + \alpha r_{ui},

with :math:`\alpha = 40` reported as working well. Every pair keeps
some minimal confidence, which matters because the zeros are not
negatives: a user may not have consumed an item through ignorance,
price or availability rather than dislike. Symmetrically, consumption
is not proof of preference -- a gift, or the channel left on after the
previous show.

**The cost function therefore ranges over every pair**, not only the
observed ones:

.. math:: \min_{x,y} \sum_{u,i} c_{ui}(p_{ui} - x_u^\top y_i)^2
          + \lambda\Big(\sum_u \|x_u\|^2 + \sum_i \|y_i\|^2\Big).

That is :math:`m \cdot n` terms -- billions in practice -- which rules
out stochastic gradient descent over the observed entries and is why
the method is alternating least squares.

**The trick that makes ALS linear in the input.** The user update is
:math:`x_u = (Y^\top C^u Y + \lambda I)^{-1}Y^\top C^u p(u)`, whose
naive cost is :math:`O(f^2 n)` per user. Writing
:math:`Y^\top C^u Y = Y^\top Y + Y^\top (C^u - I)Y` moves the
expensive part into a term that is precomputed once, leaving a
correction whose :math:`C^u - I` has only :math:`n_u` non-zeros. Cost
falls to :math:`O(f^2 n_u + f^3)` per user and :math:`O(f^2N + f^3m)`
overall -- linear in the data. The decomposition is implemented as
written, and the anchor checks it against the naive form.

**Explanations fall out of the same algebra.** Substituting the user
update into :math:`\hat p_{ui} = y_i^\top x_u` gives
:math:`y_i^\top W^u Y^\top C^u p(u)` with
:math:`W^u = (Y^\top C^u Y + \lambda I)^{-1}`, so the prediction
decomposes into per-item contributions from the user's own history --
a property latent factor models usually lack, because the factors
block any direct link between past actions and output.

References
----------
Hu, Y., Koren, Y. & Volinsky, C. (2008) "Collaborative Filtering for
Implicit Feedback Datasets", *Proceedings of the Eighth IEEE
International Conference on Data Mining (ICDM 2008)*, 263-272,
doi:10.1109/ICDM.2008.22. Sec. 2 (the four properties of implicit
feedback, in particular that its numerical value indicates confidence
rather than preference). Sec. 4 (the binarised p_ui, the confidence
c_ui = 1 + alpha r_ui with alpha = 40, the cost function of eq. (3)
summed over ALL m*n pairs, the alternating least squares update
eq. (4), and the Y'C^u Y = Y'Y + Y'(C^u - I)Y decomposition giving
O(f^2 N + f^3 m) total time). Sec. 5 (explaining recommendations
through W^u).

Koren, Y., Bell, R. & Volinsky, C. (2009) "Matrix Factorization
Techniques for Recommender Systems", *Computer* 42(8), 30-37,
doi:10.1109/MC.2009.263. The explicit-feedback factorisation this is
adapted from.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["preference", "confidence", "als_step", "fit_wrmf",
           "cost", "explain"]

_EPS = 1e-12


def preference(r):
    r""":math:`p_{ui} = 1` if :math:`r_{ui} > 0`, else 0."""
    return [[1.0 if float(v) > 0.0 else 0.0 for v in row]
            for row in k.mat(r)]


def confidence(r, alpha=40.0):
    r""":math:`c_{ui} = 1 + \alpha r_{ui}` -- every pair keeps some."""
    a = float(alpha)
    if a < 0.0:
        raise ValueError("impFB: alpha must be non-negative")
    return [[1.0 + a * float(v) for v in row] for row in k.mat(r)]


def _solve(A, b):
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) < 1e-14:
            raise ValueError("impFB: the normal equations are "
                             "singular; increase lambda")
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0.0:
                f = M[i][c]
                M[i] = [M[i][j] - f * M[c][j] for j in range(n + 1)]
    return [M[i][n] for i in range(n)]


def als_step(Y, C_row, p_row, lam, fast=True):
    r"""Eq. (4) for one user.

    ``fast=True`` uses :math:`Y^\top Y + Y^\top(C^u - I)Y`, touching
    only the user's :math:`n_u` non-zeros; ``fast=False`` forms
    :math:`Y^\top C^u Y` directly. Both must agree exactly.
    """
    n, f = len(Y), len(Y[0])
    lm = float(lam)
    if fast:
        A = [[sum(Y[i][a] * Y[i][b] for i in range(n))
              for b in range(f)] for a in range(f)]
        nz = [i for i in range(n) if C_row[i] != 1.0]
        for i in nz:
            w = C_row[i] - 1.0
            for a in range(f):
                for b in range(f):
                    A[a][b] += w * Y[i][a] * Y[i][b]
    else:
        A = [[sum(C_row[i] * Y[i][a] * Y[i][b] for i in range(n))
              for b in range(f)] for a in range(f)]
    for a in range(f):
        A[a][a] += lm
    rhs = [sum(C_row[i] * p_row[i] * Y[i][a] for i in range(n))
           for a in range(f)]
    return _solve(A, rhs)


def cost(R, X, Y, alpha=40.0, lam=0.1):
    r"""Eq. (3), summed over EVERY user-item pair."""
    P = preference(R)
    C = confidence(R, alpha)
    tot = 0.0
    for u in range(len(P)):
        for i in range(len(P[0])):
            e = P[u][i] - sum(X[u][f] * Y[i][f]
                              for f in range(len(X[0])))
            tot += C[u][i] * e * e
    reg = sum(v * v for r in X for v in r) + \
        sum(v * v for r in Y for v in r)
    return tot + float(lam) * reg


def fit_wrmf(R, f=8, alpha=40.0, lam=0.1, iters=15, seed=0,
             fast=True):
    r"""Alternating least squares over all pairs."""
    M = [[float(v) for v in row] for row in k.mat(R)]
    m, n = len(M), len(M[0])
    if int(f) < 1:
        raise ValueError("impFB: f must be at least 1")
    if any(v < 0.0 for r in M for v in r):
        raise ValueError("impFB: implicit counts cannot be negative")
    P, C = preference(M), confidence(M, alpha)
    rng = np.random.default_rng(seed)
    X = [[float(rng.uniform()) * 0.1 for _ in range(int(f))]
         for _ in range(m)]
    Y = [[float(rng.uniform()) * 0.1 for _ in range(int(f))]
         for _ in range(n)]
    hist = []
    for _ in range(int(iters)):
        for u in range(m):
            X[u] = als_step(Y, C[u], P[u], lam, fast)
        for i in range(n):
            col_c = [C[u][i] for u in range(m)]
            col_p = [P[u][i] for u in range(m)]
            Y[i] = als_step(X, col_c, col_p, lam, fast)
        hist.append(cost(M, X, Y, alpha, lam))
    return RichResult(payload={
        "estimate": (X, Y), "X": X, "Y": Y, "cost_history": hist,
        "final_cost": hist[-1] if hist else float("nan"),
        "f": int(f), "alpha": float(alpha), "lambda": float(lam),
        "method": "weighted ALS over all m*n pairs; Hu, Koren & "
                  "Volinsky (2008) eqs. (3)-(4)",
        "note": "the numerical value of implicit feedback is "
                "CONFIDENCE, not preference",
    })


def explain(Y, C_row, p_row, i, lam=0.1):
    r"""Sec. 5: decompose :math:`\hat p_{ui}` over the user's history.

    :math:`\hat p_{ui} = y_i^\top W^u Y^\top C^u p(u)`, so each past
    item contributes a term.
    """
    n, f = len(Y), len(Y[0])
    A = [[sum(C_row[t] * Y[t][a] * Y[t][b] for t in range(n))
          for b in range(f)] for a in range(f)]
    for a in range(f):
        A[a][a] += float(lam)
    W = [_solve(A, [1.0 if b == a else 0.0 for b in range(f)])
         for a in range(f)]
    yi = Y[int(i)]
    v = [sum(W[a][b] * yi[b] for b in range(f)) for a in range(f)]
    terms = {j: C_row[j] * p_row[j]
             * sum(v[a] * Y[j][a] for a in range(f))
             for j in range(n) if p_row[j] > 0.0}
    return {"contributions": terms, "prediction": sum(terms.values()),
            "note": "each past item's share of the predicted "
                    "preference"}


def cheatsheet():
    return ("impFB: implicit feedback measures CONFIDENCE, not "
            "preference -- the favourite film is watched once, the "
            "merely-liked series weekly. Split into binary p_ui and "
            "c_ui = 1 + alpha r_ui (alpha = 40). The cost sums over "
            "ALL m*n pairs, because zeros are missing evidence rather "
            "than negatives, which rules out SGD and forces ALS. "
            "Y'C^u Y = Y'Y + Y'(C^u - I)Y makes each update "
            "O(f^2 n_u + f^3), linear in the input. Substituting the "
            "update into the prediction yields per-item explanations.")


# compact alias per ledger/NAMING.md
implicitfeedback = fit_wrmf

# public names resolved by fn/_lazy_map.json
implicit_feedback_loss = fit_wrmf
