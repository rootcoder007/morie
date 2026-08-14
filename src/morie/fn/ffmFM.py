# morie.fn -- function file (rootcoder007/morie)
r"""Field-aware factorization machines.

An FM gives each feature one latent vector, used against every other
feature. FFM's observation is that a feature interacts differently
depending on *what kind* of feature it is meeting: the latent vector
for "ESPN" should differ when it is paired with an advertiser than
when it is paired with a gender. So each feature gets one vector per
**field**:

.. math:: \phi(w, x) = \sum_{j_1=1}^{n}\sum_{j_2=j_1+1}^{n}
          \langle w_{j_1, f_2},\, w_{j_2, f_1}\rangle x_{j_1}x_{j_2},

where :math:`f_1, f_2` are the fields of features :math:`j_1, j_2`.
Note the crossing: the vector used for :math:`j_1` is the one indexed
by the *other* feature's field. Getting that backwards is the classic
FFM implementation bug, and the anchor pins it with an asymmetric
construction that a self-field indexing would score differently.

**The cost of the extra index.** FM has :math:`nk` parameters; FFM has
:math:`nfk`. Since each vector now specialises to one field, it needs
less capacity, and in practice :math:`k` for FFM is much smaller than
for FM. There is no linear-time reformulation: the FM identity relies
on one vector per feature, so FFM's evaluation is
:math:`O(\bar n^2 k)` over non-zeros.

**Training details that are not incidental.** Logistic loss with
:math:`y \in \{-1, 1\}`, per-coordinate AdaGrad accumulating squared
gradients :math:`(G_{j,f})_d`, and -- because the model overfits
readily -- early stopping on a validation set, with the paper noting
that one epoch was enough in the original competition use.

References
----------
Juan, Y., Zhuang, Y., Chin, W.-S. & Lin, C.-J. (2016) "Field-aware
Factorization Machines for CTR Prediction", *Proceedings of the Tenth
ACM Conference on Recommender Systems (RecSys '16)*, 43-50,
doi:10.1145/2959100.2959134. Sec. 2 (FM and its limitation), Sec. 3
(the FFM model equation with the crossed field indices
<w_{j1,f2}, w_{j2,f1}>, the nfk parameter count and the consequence
that FFM's k is much smaller than FM's, the logistic loss with
y in {-1,1}, and the AdaGrad updates of eqs. (8)-(9)), and Sec. 3.3
(overfitting and early stopping).

Rendle, S. (2010) "Factorization Machines", *ICDM 2010*, 995-1000,
doi:10.1109/ICDM.2010.127. The model FFM specialises; implemented in
:mod:`fmFM`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["phi", "logistic_loss", "fit_ffm", "n_parameters"]

_EPS = 1e-12


def n_parameters(n_features, n_fields, k_dim, model="ffm"):
    r""":math:`nfk` for FFM against :math:`nk` for FM."""
    n, f, kk = int(n_features), int(n_fields), int(k_dim)
    if model not in ("ffm", "fm"):
        raise ValueError("ffmFM: model must be ffm or fm, got %r"
                         % (model,))
    return n * f * kk if model == "ffm" else n * kk


def phi(x, fields, W):
    r"""The FFM interaction term, with the field indices CROSSED.

    ``x`` is a list of (feature index, value); ``fields[j]`` is the
    field of feature ``j``; ``W[j][f]`` is feature ``j``'s vector for
    field ``f``.
    """
    nz = [(int(j), float(v)) for j, v in x if float(v) != 0.0]
    tot = 0.0
    for a in range(len(nz)):
        for b in range(a + 1, len(nz)):
            j1, v1 = nz[a]
            j2, v2 = nz[b]
            f1, f2 = int(fields[j1]), int(fields[j2])
            tot += sum(W[j1][f2][d] * W[j2][f1][d]
                       for d in range(len(W[j1][f2]))) * v1 * v2
    return tot


def logistic_loss(y, phi_val):
    r""":math:`\log(1 + e^{-y\phi})`, with :math:`y \in \{-1,1\}`."""
    yv = float(y)
    if yv not in (-1.0, 1.0):
        raise ValueError("ffmFM: the label must be -1 or 1, got %r"
                         % (y,))
    z = -yv * float(phi_val)
    return math.log(1.0 + math.exp(z)) if z < 700 else z


def fit_ffm(rows, labels, fields, n_features, n_fields, k_dim=4,
            eta=0.1, lam=2e-5, epochs=10, seed=0):
    r"""AdaGrad SGD on the logistic loss, per eqs. (8)-(9)."""
    n, F, kk = int(n_features), int(n_fields), int(k_dim)
    if n < 1 or F < 1 or kk < 1:
        raise ValueError("ffmFM: n_features, n_fields and k must all "
                         "be at least 1")
    if len(rows) != len(labels):
        raise ValueError("ffmFM: %d rows but %d labels"
                         % (len(rows), len(labels)))
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(kk)
    W = [[[float(rng.uniform()) * scale for _ in range(kk)]
          for _ in range(F)] for _ in range(n)]
    G = [[[1.0] * kk for _ in range(F)] for _ in range(n)]
    hist = []
    for ep in range(int(epochs)):
        tot = 0.0
        for r in range(len(rows)):
            y = float(labels[r])
            p = phi(rows[r], fields, W)
            tot += logistic_loss(y, p)
            g0 = -y / (1.0 + math.exp(min(700.0, y * p)))
            nz = [(int(j), float(v)) for j, v in rows[r]
                  if float(v) != 0.0]
            for a in range(len(nz)):
                for b in range(a + 1, len(nz)):
                    j1, v1 = nz[a]
                    j2, v2 = nz[b]
                    f1, f2 = int(fields[j1]), int(fields[j2])
                    for d in range(kk):
                        g1 = lam * W[j1][f2][d] \
                            + g0 * W[j2][f1][d] * v1 * v2
                        g2 = lam * W[j2][f1][d] \
                            + g0 * W[j1][f2][d] * v1 * v2
                        G[j1][f2][d] += g1 * g1
                        G[j2][f1][d] += g2 * g2
                        W[j1][f2][d] -= eta / math.sqrt(G[j1][f2][d]) \
                            * g1
                        W[j2][f1][d] -= eta / math.sqrt(G[j2][f1][d]) \
                            * g2
        hist.append(tot / len(rows))
    return RichResult(payload={
        "estimate": W, "W": W, "loss_history": hist,
        "final_loss": hist[-1], "k": kk,
        "n_parameters": n_parameters(n, F, kk),
        "n_parameters_fm": n_parameters(n, F, kk, "fm"),
        "method": "FFM with AdaGrad; Juan, Zhuang, Chin & Lin (2016) "
                  "eqs. (8)-(9)",
        "caveat": "FFM overfits readily -- the paper stops early on a "
                  "validation set",
    })


def cheatsheet():
    return ("ffmFM: one latent vector per feature PER FIELD, because a "
            "feature interacts differently with an advertiser than "
            "with a gender. The interaction is "
            "<w_{j1,f2}, w_{j2,f1}> -- each vector indexed by the "
            "OTHER feature's field, and swapping that is the classic "
            "bug. nfk parameters against FM's nk, but each vector "
            "specialises so k is much smaller. No linear-time trick: "
            "the FM identity needs one vector per feature. Logistic "
            "loss with y in {-1,1}, AdaGrad, early stopping.")


# compact alias per ledger/NAMING.md
fieldawarefm = fit_ffm
