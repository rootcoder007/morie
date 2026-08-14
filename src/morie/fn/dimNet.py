# morie.fn -- function file (rootcoder007/morie)
r"""DimeNet: directional message passing.

A distance-only graph network represents a molecule by internuclear
distances alone. Empirical potentials do not:

.. math:: E = E_{\text{bonds}} + E_{\text{angle}}
          + E_{\text{torsion}} + E_{\text{non-bonded}},

and the angular term is not recoverable from distances between *pairs*
-- three atoms in a line and three at a right angle can share the same
two bond lengths. Directional information plays a central role in
those potentials, and a pairwise-distance model cannot express it.

**Embed the messages, not the atoms.** Each message
:math:`m_{ji}` is associated with a direction in coordinate space, so
directional message embeddings are **rotationally equivariant**: the
directions rotate with the molecule. Messages are then updated using
the **angle between them**, in a scheme analogous to belief
propagation,

.. math:: m_{ji}^{(l+1)} = f\Big(m_{ji}^{(l)},
          \sum_{k \in N_j \setminus \{i\}}
          f_{\text{int}}\big(m_{kj}^{(l)}, e_{RBF}(d_{kj}),
          a_{SBF}(\alpha_{kji})\big)\Big),

where :math:`\alpha_{kji}` is the angle at :math:`j` between the
incoming and outgoing bonds. That angle is the quantity a distance-only
model never sees.

**The basis is chosen, not assumed.** Spherical Bessel functions
radially and spherical harmonics angularly give theoretically
well-founded, **orthogonal** representations; the paper reports they
outperform the prevalent Gaussian radial basis while using fewer than
a quarter of the parameters. Orthogonality is the reason -- a
non-orthogonal basis spends parameters representing the same thing
twice.

**Cost.** Messages live on directed *edges* and interact over
angles, so the interaction is over **triplets**: the work scales with
the number of angles, not the number of pairs, and
``triplet_count`` reports it so the cost is explicit.

References
----------
Klicpera, J., Gross, J. & Gunnemann, S. (2020) "Directional Message
Passing for Molecular Graphs", *International Conference on Learning
Representations (ICLR 2020)*, arXiv:2003.03123. The abstract: existing
models represent a molecule as a graph using only the distance between
atoms and do not consider the spatial direction from one atom to
another, despite directional information playing a central role in
empirical potentials, e.g. angular potentials; directional message
passing embedding the MESSAGES rather than the atoms, each associated
with a direction in coordinate space and hence rotationally
equivariant; a message passing scheme analogous to belief propagation
that transforms messages based on the ANGLE between them; spherical
Bessel functions and spherical harmonics giving theoretically
well-founded ORTHOGONAL representations that outperform the prevalent
Gaussian radial basis while using fewer than 1/4 of the parameters;
and the decomposition of classical empirical potentials into bond,
angle, torsion and non-bonded terms.

Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. & Dahl, G. E.
(2017) "Neural Message Passing for Quantum Chemistry", *ICML 2017*,
PMLR 70, 1263-1272, arXiv:1704.01212. Implemented in :mod:`mpfn`.

Schutt, K. T. et al. (2017) "SchNet: A continuous-filter convolutional
neural network for modeling quantum interactions", *NeurIPS 2017*,
arXiv:1706.08566. The distance-only predecessor; implemented in
:mod:`schN`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["angle_between", "triplet_count", "bessel_basis",
           "spherical_harmonic_basis", "directional_message_pass"]

_EPS = 1e-12


def angle_between(r_k, r_j, r_i):
    r"""The angle :math:`\alpha_{kji}` at :math:`j`.

    Two configurations can share every pairwise distance in the
    message's own pair and differ here -- which is precisely what a
    distance-only model cannot see.
    """
    a = [float(v) for v in k.vec(r_k)]
    b = [float(v) for v in k.vec(r_j)]
    c = [float(v) for v in k.vec(r_i)]
    u = [a[i] - b[i] for i in range(len(a))]
    v = [c[i] - b[i] for i in range(len(a))]
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu <= _EPS or nv <= _EPS:
        raise ValueError("dimNet: an angle needs three distinct "
                         "positions")
    cs = sum(u[i] * v[i] for i in range(len(u))) / (nu * nv)
    return math.acos(min(max(cs, -1.0), 1.0))


def triplet_count(adj):
    r"""Angles, not pairs -- the quantity the cost actually scales
    with."""
    n = 0
    for j in adj:
        d = len(set(adj[j]) - {j})
        n += d * (d - 1)
    return {"triplets": n,
            "pairs": sum(len(set(adj[j]) - {j}) for j in adj),
            "note": "directional message passing interacts over "
                    "TRIPLETS; the cost follows the angle count"}


def bessel_basis(d, cutoff=5.0, n_basis=8):
    r"""Spherical Bessel radial basis:
    :math:`\sqrt{2/c}\,\frac{\sin(n\pi d/c)}{d}`.

    Orthogonal on :math:`[0,c]`, which is why it needs far fewer
    parameters than a Gaussian basis to say the same thing.
    """
    c = float(cutoff)
    if c <= 0.0:
        raise ValueError("dimNet: the cutoff must be positive")
    dv = float(d)
    if dv <= 0.0:
        raise ValueError("dimNet: the distance must be positive")
    return [math.sqrt(2.0 / c) * math.sin(n * math.pi * dv / c) / dv
            for n in range(1, int(n_basis) + 1)]


def spherical_harmonic_basis(angle, n_basis=4):
    r"""Angular basis: Legendre polynomials in :math:`\cos\alpha`.

    The :math:`m = 0` spherical harmonics, orthogonal on the sphere
    -- the angular counterpart of the Bessel radial basis.
    """
    x = math.cos(float(angle))
    n = int(n_basis)
    if n < 1:
        raise ValueError("dimNet: at least one basis function is "
                         "needed")
    out = [1.0]
    if n > 1:
        out.append(x)
    for l in range(2, n):
        out.append(((2 * l - 1) * x * out[l - 1]
                    - (l - 1) * out[l - 2]) / l)
    return out[:n]


def directional_message_pass(messages, adj, R, interact, update,
                             cutoff=5.0, n_rbf=8, n_sbf=4):
    r"""One round of directional message passing.

    ``messages[(j,i)]`` is the embedding of the directed edge
    :math:`j \to i`; each is updated from the incoming messages
    :math:`k \to j` with :math:`k \ne i`, transformed by the angle
    between them.
    """
    pos = [[float(v) for v in r] for r in k.mat(R)]
    out = {}
    for (j, i) in messages:
        acc = None
        for kk in sorted(set(adj.get(j, ())) - {i, j}):
            d = math.sqrt(sum((pos[kk][a] - pos[j][a]) ** 2
                              for a in range(len(pos[j]))))
            ang = angle_between(pos[kk], pos[j], pos[i])
            contrib = interact(messages[(kk, j)],
                               bessel_basis(d, cutoff, n_rbf),
                               spherical_harmonic_basis(ang, n_sbf))
            contrib = [float(v) for v in contrib]
            acc = list(contrib) if acc is None else \
                [acc[a] + contrib[a] for a in range(len(acc))]
        if acc is None:
            acc = [0.0] * len(messages[(j, i)])
        out[(j, i)] = [float(v) for v in update(messages[(j, i)], acc)]
    return RichResult(payload={
        "estimate": out, "messages": out,
        "n_messages": len(out),
        "triplets": triplet_count(adj)["triplets"],
        "method": "directional message passing; Klicpera, Gross & "
                  "Gunnemann (2020)",
        "note": "messages carry DIRECTION, so they are rotationally "
                "equivariant, and interact through the ANGLE between "
                "them",
    })


def cheatsheet():
    return ("dimNet: a distance-only graph network cannot express the "
            "ANGULAR term of an empirical potential -- three atoms in "
            "a line and three at a right angle can share the same bond "
            "lengths. So embed the MESSAGES, not the atoms: each "
            "carries a direction, hence is rotationally EQUIVARIANT, "
            "and messages interact through the angle between them, "
            "belief-propagation style. The basis is spherical BESSEL "
            "radially and spherical HARMONICS angularly -- orthogonal, "
            "so it beats a Gaussian basis with under a quarter of the "
            "parameters. Cost scales with TRIPLETS, not pairs.")


# compact alias per ledger/NAMING.md
dimenet = directional_message_pass
