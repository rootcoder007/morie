# morie.fn -- function file (rootcoder007/morie)
r"""PaiNN: equivariant message passing for tensorial properties.

Message passing networks scale to large training sets but had proven
**less data efficient** than kernel methods. PaiNN's diagnosis is
specific: the limitation is the use of **invariant** representations.
A network whose every internal feature is a scalar can only combine
distances, so it needs many more examples to learn what a directional
representation gets for free -- and it cannot predict a tensor at all
without bolting on a separate head.

**Two kinds of feature, updated together.** Each atom carries a scalar
:math:`s_i \in \mathbb{R}^F` and a **vector**
:math:`\vec v_i \in \mathbb{R}^{3\times F}`. The message and update
blocks mix them under rules that preserve type:

* scalar :math:`\times` scalar, and :math:`\|\vec v\|`, give scalars;
* scalar :math:`\times` vector, and :math:`s\,\hat r_{ij}`, give
  vectors;
* :math:`\vec v_1 \cdot \vec v_2` gives a **scalar** -- an invariant
  built from two equivariant quantities, which is how directional
  information re-enters the scalar channel.

Rotate the molecule and every scalar is unchanged while every vector
rotates with it. ``equivariance_error`` checks both at once, because
checking only the scalars would pass a model that has silently lost
its vectors.

**Tensorial properties come out directly.** Dipole moments,
polarizabilities and the like are tensors; with equivariant atomwise
representations they are read off rather than approximated, which is
what enables the molecular-spectra application and its reported
speed-up of four to five orders of magnitude over the electronic
structure reference.

**Smaller, not larger.** The paper reports improved benchmarks with
*reduced* model size and inference time -- equivariance is not bought
with parameters here; it replaces them.

References
----------
Schutt, K. T., Unke, O. T. & Gastegger, M. (2021) "Equivariant message
passing for the prediction of tensorial properties and molecular
spectra", *Proceedings of the 38th International Conference on Machine
Learning (ICML 2021)*, PMLR 139, 9377-9388, arXiv:2102.03150. The
abstract: message passing networks scale readily to large training
sets but have proven less data efficient than kernel methods; the
identification of the limitations of INVARIANT representations as a
major reason; the extension of message passing to rotationally
EQUIVARIANT representations; the polarizable atom interaction neural
network improving on common molecule benchmarks while REDUCING model
size and inference time; and the use of equivariant atomwise
representations for tensorial properties and molecular spectra, with
speedups of 4-5 orders of magnitude over the electronic structure
reference.

Schutt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S.,
Tkatchenko, A. & Muller, K.-R. (2017) "SchNet", *NeurIPS 2017*,
arXiv:1706.08566. The invariant predecessor; implemented in
:mod:`schN`.

Satorras, V. G., Hoogeboom, E. & Welling, M. (2021) "E(n) Equivariant
Graph Neural Networks", *ICML 2021*, PMLR 139, 9323-9332,
arXiv:2102.09844. Implemented in :mod:`egnnL`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["scalar_vector_message", "gated_update", "vector_norm",
           "equivariance_error", "dipole_moment"]

_EPS = 1e-12


def vector_norm(v):
    r""":math:`\|\vec v\|` -- an invariant built from an equivariant."""
    a = [[float(q) for q in r] for r in k.mat(v)]
    return [math.sqrt(sum(a[d][f] ** 2 for d in range(len(a))))
            for f in range(len(a[0]))]


def scalar_vector_message(s_j, v_j, r_ij, phi_s, phi_v, W_rbf):
    r"""The message: scalars from scalars, vectors from
    :math:`s\,\hat r` and :math:`s\,\vec v`.

    Type is preserved by construction, which is what makes the whole
    network equivariant rather than approximately so.
    """
    s = [float(q) for q in k.vec(s_j)]
    V = [[float(q) for q in r] for r in k.mat(v_j)]
    r = [float(q) for q in k.vec(r_ij)]
    d = math.sqrt(sum(q * q for q in r))
    if d <= _EPS:
        raise ValueError("painn: two atoms occupy the same position")
    hat = [q / d for q in r]
    w = [float(q) for q in W_rbf(d)]
    ds = [float(q) for q in phi_s(s, w)]
    dv_scale = [float(q) for q in phi_v(s, w)]
    F = len(s)
    if len(ds) != F or len(dv_scale) != 2 * F:
        raise ValueError("painn: the message networks are mis-sized "
                         "(need F scalars and 2F vector gates)")
    dv = [[dv_scale[f] * V[a][f]
           + dv_scale[F + f] * hat[a] for f in range(F)]
          for a in range(len(hat))]
    return {"ds": ds, "dv": dv,
            "note": "scalar*vector and s*r_hat give VECTORS; nothing "
                    "mixes the types"}


def gated_update(s, v, U, V, phi):
    r"""The update block: :math:`\vec v_1\cdot\vec v_2` re-enters the
    scalars.

    That inner product is the only path from the vector channel back
    to the scalar one, and it is invariant -- which is why the network
    can use directional information without breaking invariance of the
    energy.
    """
    sv = [float(q) for q in k.vec(s)]
    Vv = [[float(q) for q in r] for r in k.mat(v)]
    D, F = len(Vv), len(Vv[0])
    Uv = [[sum(U[f][g] * Vv[a][g] for g in range(F))
           for f in range(F)] for a in range(D)]
    Vw = [[sum(V[f][g] * Vv[a][g] for g in range(F))
           for f in range(F)] for a in range(D)]
    dot = [sum(Uv[a][f] * Vw[a][f] for a in range(D))
           for f in range(F)]
    nrm = vector_norm(Vw)
    out = phi(sv, dot, nrm)
    ds = [float(q) for q in out["ds"]]
    gate = [float(q) for q in out["gate"]]
    if len(ds) != F or len(gate) != F:
        raise ValueError("painn: the update network is mis-sized")
    dv = [[gate[f] * Uv[a][f] for f in range(F)] for a in range(D)]
    return {"ds": ds, "dv": dv, "scalar_from_vectors": dot,
            "note": "the vector-vector inner product is the ONLY path "
                    "back to the scalar channel, and it is invariant"}


def dipole_moment(charges, R, centre=None):
    r""":math:`\vec\mu = \sum_i q_i (\vec r_i - \vec r_c)`.

    A tensorial (here vector) property, read off directly from
    equivariant atomwise quantities rather than approximated.
    """
    q = [float(v) for v in k.vec(charges)]
    pos = [[float(v) for v in r] for r in k.mat(R)]
    if len(q) != len(pos):
        raise ValueError("painn: %d charges but %d positions"
                         % (len(q), len(pos)))
    d = len(pos[0])
    c = [sum(p[a] for p in pos) / len(pos) for a in range(d)] \
        if centre is None else [float(v) for v in k.vec(centre)]
    mu = [sum(q[i] * (pos[i][a] - c[a]) for i in range(len(q)))
          for a in range(d)]
    return {"dipole": mu,
            "magnitude": math.sqrt(sum(v * v for v in mu)),
            "note": "a VECTOR property; an invariant network cannot "
                    "produce one without a separate head"}


def equivariance_error(model, s, v, R, Q, tol=1e-9):
    r"""Rotate the input; scalars must not move, vectors must rotate.

    Checking only the scalars would pass a model that has silently
    lost its vector channel.
    """
    pos = [[float(q) for q in r] for r in k.mat(R)]
    d = len(pos[0])
    rot_R = [[sum(Q[a][b] * pos[i][b] for b in range(d))
              for a in range(d)] for i in range(len(pos))]
    V = [[float(q) for q in r] for r in k.mat(v)]
    rot_v = [[sum(Q[a][b] * V[b][f] for b in range(d))
              for f in range(len(V[0]))] for a in range(d)]
    base = model(s, V, pos)
    other = model(s, rot_v, rot_R)
    se = max(abs(float(base["s"][f]) - float(other["s"][f]))
             for f in range(len(base["s"])))
    want = [[sum(Q[a][b] * float(base["v"][b][f]) for b in range(d))
             for f in range(len(base["v"][0]))] for a in range(d)]
    ve = max(abs(float(other["v"][a][f]) - want[a][f])
             for a in range(d) for f in range(len(want[0])))
    return {"scalar_error": se, "vector_error": ve,
            "scalars_invariant": se < float(tol),
            "vectors_equivariant": ve < float(tol),
            "note": "both must hold; checking only the scalars passes "
                    "a model that has lost its vectors"}


def cheatsheet():
    return ("painn: message passing was LESS DATA EFFICIENT than "
            "kernel methods, and the diagnosis is INVARIANT "
            "representations -- a network of scalars can only combine "
            "distances and cannot emit a tensor at all. Carry BOTH a "
            "scalar and a VECTOR feature per atom and preserve type: "
            "s*s and ||v|| give scalars, s*v and s*r_hat give vectors, "
            "and v1.v2 is the ONLY route back from vectors to scalars "
            "-- invariant, so the energy stays invariant while "
            "direction is used. Tensorial properties are read off "
            "directly, and the model is SMALLER, not larger.")


# compact alias per ledger/NAMING.md
painnnet = gated_update

# public names resolved by fn/_lazy_map.json
painn = gated_update
