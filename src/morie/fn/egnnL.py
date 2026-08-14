# morie.fn -- function file (rootcoder007/morie)
r"""E(n)-equivariant graph neural networks.

For systems with geometry -- molecules, point clouds, n-body dynamics
-- the physics does not change when the whole configuration is
translated, rotated or reflected. A model that respects this by
construction is stated as

.. math:: Q x^{l+1} + g,\; h^{l+1} = \mathrm{EGCL}(Qx^l + g,\; h^l)

for every orthogonal :math:`Q` and every translation :math:`g`. The
usual route to that guarantee is spherical harmonics and higher-order
representations. This paper's point is that none of that is needed.

**The layer, in four equations.**

.. math:: m_{ij} &= \phi_e\big(h_i^l, h_j^l,
            \|x_i^l - x_j^l\|^2, a_{ij}\big) \\
          x_i^{l+1} &= x_i^l + C\sum_{j \ne i}
            (x_i^l - x_j^l)\,\phi_x(m_{ij}) \\
          m_i &= \sum_{j \ne i} m_{ij} \\
          h_i^{l+1} &= \phi_h(h_i^l, m_i)

**Why this is equivariant, in one line each.** The message depends on
positions only through the *squared distance*, which no rotation,
reflection or translation changes -- so :math:`m_{ij}` is invariant.
The coordinate update adds a weighted sum of relative differences
:math:`x_i - x_j`, which transforms as a vector: rotate the input and
the update rotates with it; translate and the differences are
unchanged. Equations 5 and 6 touch only invariant quantities, so
:math:`h^{l+1}` stays invariant. Composing layers preserves both, by
induction.

**Equation 4 is the whole difference from a standard GNN**, and
:math:`C = 1/(M-1)` simply averages the sum. Despite that simplicity
the operation is flexible, because :math:`m_{ij}` may carry information
from the entire graph rather than just the edge.

**Momentum, when velocity matters.** Replacing equation 4 with a
velocity-carrying update keeps an explicit momentum estimate at every
layer, which also allows a non-zero initial velocity. Both routes are
implemented.

References
----------
Satorras, V. G., Hoogeboom, E. & Welling, M. (2021) "E(n) Equivariant
Graph Neural Networks", *Proceedings of the 38th International
Conference on Machine Learning (ICML 2021)*, PMLR 139, 9323-9332,
arXiv:2102.09844. Sec. 3 (the EGCL of eqs. (3)-(6), with C = 1/(M-1);
the statement that eq. (4) is the main difference from standard GNNs
and the reason equivariances 1 and 2 are preserved). Sec. 3.1 (the
equivariance condition Qx + g; that m_ij is E(n) invariant because it
depends on positions only through squared distances; that the weighted
sum of differences transforms as a type-1 vector; and the inductive
argument for composed layers). Sec. 3.2 (the momentum variant
replacing eq. (4)).

Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K. &
Riley, P. (2018) "Tensor Field Networks: Rotation- and
Translation-Equivariant Neural Networks for 3D Point Clouds",
arXiv:1802.08219. The higher-order-representation approach this
avoids.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["edge_message", "coord_update", "egcl", "run_egnn",
           "equivariance_error"]

_EPS = 1e-12
_MODES = ("position", "momentum")


def _sqdist(a, b):
    return sum((a[d] - b[d]) ** 2 for d in range(len(a)))


def edge_message(h_i, h_j, x_i, x_j, phi_e, a_ij=None):
    r"""Eq. (3). Positions enter ONLY as :math:`\|x_i - x_j\|^2`,
    which is what makes the message invariant."""
    return phi_e(list(h_i), list(h_j), _sqdist(x_i, x_j), a_ij)


def coord_update(X, M, phi_x, C=None):
    r"""Eq. (4): :math:`x_i + C\sum_j (x_i - x_j)\phi_x(m_{ij})`."""
    n = len(X)
    if n < 2:
        raise ValueError("egnnL: need at least 2 particles")
    c = 1.0 / (n - 1) if C is None else float(C)
    out = []
    for i in range(n):
        acc = list(X[i])
        for j in range(n):
            if j == i:
                continue
            w = float(phi_x(M[i][j]))
            acc = [acc[d] + c * (X[i][d] - X[j][d]) * w
                   for d in range(len(acc))]
        out.append(acc)
    return out


def egcl(H, X, phi_e, phi_x, phi_h, A=None, C=None, V=None,
         mode="position", phi_v=None, dt=1.0):
    r"""One equivariant graph convolutional layer, eqs. (3)-(6)."""
    if mode not in _MODES:
        raise ValueError("egnnL: mode must be one of %s, got %r"
                         % (", ".join(_MODES), mode))
    n = len(H)
    M = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                a = None if A is None else A.get((i, j))
                M[i][j] = edge_message(H[i], H[j], X[i], X[j], phi_e, a)
    if mode == "position":
        Xn = coord_update(X, M, phi_x, C)
        Vn = V
    else:
        if V is None or phi_v is None:
            raise ValueError("egnnL: the momentum variant needs V and "
                             "phi_v")
        c = 1.0 / (n - 1) if C is None else float(C)
        Vn = []
        for i in range(n):
            acc = [float(phi_v(H[i])) * V[i][d]
                   for d in range(len(V[i]))]
            for j in range(n):
                if j == i:
                    continue
                w = float(phi_x(M[i][j]))
                acc = [acc[d] + c * (X[i][d] - X[j][d]) * w
                       for d in range(len(acc))]
            Vn.append(acc)
        Xn = [[X[i][d] + dt * Vn[i][d] for d in range(len(X[i]))]
              for i in range(n)]
    Hn = []
    for i in range(n):
        mi = None
        for j in range(n):
            if j == i:
                continue
            mi = list(M[i][j]) if mi is None else \
                [mi[f] + M[i][j][f] for f in range(len(mi))]
        Hn.append(phi_h(list(H[i]), mi))
    return {"H": Hn, "X": Xn, "V": Vn, "messages": M}


def run_egnn(H, X, layers, phi_e, phi_x, phi_h, A=None, C=None):
    r"""Compose layers; equivariance is preserved inductively."""
    h = [[float(v) for v in r] for r in k.mat(H)]
    x = [[float(v) for v in r] for r in k.mat(X)]
    for _ in range(int(layers)):
        r = egcl(h, x, phi_e, phi_x, phi_h, A, C)
        h, x = r["H"], r["X"]
    return RichResult(payload={
        "estimate": (h, x), "H": h, "X": x, "layers": int(layers),
        "method": "EGNN; Satorras, Hoogeboom & Welling (2021) eqs. "
                  "(3)-(6)",
        "note": "h is E(n) INVARIANT, x is E(n) EQUIVARIANT",
    })


def equivariance_error(H, X, phi_e, phi_x, phi_h, Q, g, layers=2,
                       C=None):
    r"""Transform the input, run, and compare against transforming the
    output.

    The property is stated as an equality; this measures the gap.
    """
    n, d = len(X), len(X[0])
    base = run_egnn(H, X, layers, phi_e, phi_x, phi_h, C=C)
    Xt = [[sum(Q[a][b] * X[i][b] for b in range(d)) + g[a]
           for a in range(d)] for i in range(n)]
    other = run_egnn(H, Xt, layers, phi_e, phi_x, phi_h, C=C)
    want = [[sum(Q[a][b] * base["X"][i][b] for b in range(d)) + g[a]
             for a in range(d)] for i in range(n)]
    ex = max(abs(other["X"][i][a] - want[i][a])
             for i in range(n) for a in range(d))
    eh = max(abs(other["H"][i][f] - base["H"][i][f])
             for i in range(n) for f in range(len(base["H"][0])))
    return {"coordinate_error": ex, "feature_error": eh,
            "equivariant": ex < 1e-9, "invariant": eh < 1e-9,
            "note": "x must transform WITH Q and g; h must not move at "
                    "all"}


def cheatsheet():
    return ("egnnL: equivariance to translation, rotation and "
            "reflection WITHOUT spherical harmonics. m_ij depends on "
            "position only through ||x_i - x_j||^2, so it is "
            "invariant; x_i <- x_i + C sum_j (x_i - x_j) phi_x(m_ij) "
            "adds a weighted sum of RELATIVE DIFFERENCES, which "
            "transforms as a vector. That one equation is the entire "
            "difference from a standard GNN. C = 1/(M-1). Composition "
            "preserves both properties by induction. A momentum "
            "variant replaces eq. (4) when velocity matters.")


# compact alias per ledger/NAMING.md
equivariantgnn = run_egnn

# public names resolved by fn/_lazy_map.json
egnn_layer = run_egnn
egnnlayer = run_egnn
