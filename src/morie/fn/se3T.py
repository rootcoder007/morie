# morie.fn -- function file (rootcoder007/morie)
r"""SE(3)-Transformers: attention that commutes with rotation.

For a point cloud there is no canonical orientation, so a model
without a symmetry constraint must **learn** that a rotated molecule is
the same molecule -- from data, imperfectly, and with no guarantee at
test time. Equivariance builds it in:

.. math:: f(R x + t) = R\,f(x) \quad\text{(type-1 output)},\qquad
          f(R x + t) = f(x) \quad\text{(type-0 output)},

so a transformation of the input appears as the equivalent
transformation of the output, exactly. That is a generalisation of the
translational weight-tying convolutions already have, and it restricts
the learnable function space to one that respects the task's symmetry.

**Attention makes it work on graphs of varying size**; equivariance
makes it robust. The construction is: attention *weights* are built
from **invariant** quantities -- pairwise distances and the relative
geometry -- so they are unchanged by a global rotation, while the
*values* being aggregated are equivariant. An invariant convex
combination of equivariant vectors is equivariant, which is the whole
proof and the reason ``se3_attention`` returns both.

**The anchor is the definition, not a proxy.** Rotate the input, run
the layer, and check the output equals the rotated original output to
machine precision. A model that merely *learned* rotation invariance
fails that check by a visible margin; this one cannot fail it without
a bug.

References
----------
Fuchs, F. B., Worrall, D. E., Fischer, V. & Welling, M. (2020)
"SE(3)-Transformers: 3D Roto-Translation Equivariant Attention
Networks", *Advances in Neural Information Processing Systems 33
(NeurIPS 2020)*, 1970-1981, arXiv:2006.10503. The abstract and Sec.
1-3: a variant of the self-attention module for 3D point clouds and
graphs which is equivariant under continuous 3D roto-translations;
that equivariance is important to ensure stable and predictable
performance in the presence of nuisance transformations of the input,
with increased weight-tying as a positive corollary; that
SE(3)-equivariance generalises the translational weight-tying of
conventional convolutions to roto-translations in 3D and restricts the
space of learnable functions to a subspace adhering to the symmetries
of the task; the use of self-attention to operate on point clouds and
graphs with varying numbers of points; and that the model outperforms
both a strong non-equivariant attention baseline and an equivariant
model without attention.

Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K. &
Riley, P. (2018) "Tensor field networks: Rotation- and
translation-equivariant neural networks for 3D point clouds",
arXiv:1802.08219. The equivariant kernel basis.

Vaswani, A. et al. (2017) "Attention Is All You Need", *NIPS 2017*,
5998-6008, arXiv:1706.03762. The attention module being made
equivariant.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["rotation_matrix", "invariant_features", "se3_attention",
           "check_equivariance", "radial_kernel"]

_EPS = 1e-12


def rotation_matrix(axis, angle):
    r"""Rodrigues' formula -- a genuine element of SO(3)."""
    a = [float(v) for v in k.vec(axis)]
    n = math.sqrt(sum(v * v for v in a))
    if n <= _EPS:
        raise ValueError("se3T: the rotation axis is zero")
    x, y, z = [v / n for v in a]
    c, s = math.cos(float(angle)), math.sin(float(angle))
    C = 1.0 - c
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


def _apply(R, v):
    return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def invariant_features(positions, i, j):
    r"""Relative geometry that a rotation cannot change.

    The distance and the type-0 features; the direction is kept
    separately because it is equivariant, not invariant.
    """
    P = [[float(v) for v in r] for r in k.mat(positions)]
    d = [P[j][a] - P[i][a] for a in range(3)]
    r = math.sqrt(sum(v * v for v in d))
    return {"distance": r,
            "direction": [v / r for v in d] if r > _EPS
            else [0.0, 0.0, 0.0],
            "note": "the DISTANCE is invariant; the direction is "
                    "equivariant and must not enter the weights"}


def radial_kernel(distance, weights=None, sigma=1.0):
    r"""A learnable function of the distance ALONE.

    Any dependence on absolute position or orientation here would
    destroy the equivariance, so the kernel sees only :math:`r`.
    """
    r = float(distance)
    if r < 0.0:
        raise ValueError("se3T: a distance cannot be negative")
    w = [1.0] if weights is None else [float(v) for v in k.vec(weights)]
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("se3T: sigma must be positive")
    return sum(w[m] * math.exp(-((r - m) ** 2) / (2.0 * s * s))
               for m in range(len(w)))


def se3_attention(positions, type0, type1, weights=None, sigma=1.0,
                  temperature=1.0):
    r"""Invariant attention weights over equivariant values.

    ``type0`` are scalars (invariant), ``type1`` are vectors
    (equivariant). The weights come only from distances and scalars,
    so a global rotation leaves them untouched; the aggregated vectors
    therefore rotate with the input.
    """
    P = [[float(v) for v in r] for r in k.mat(positions)]
    S = [float(v) for v in k.vec(type0)]
    V = [[float(v) for v in r] for r in k.mat(type1)]
    n = len(P)
    if len(S) != n or len(V) != n:
        raise ValueError("se3T: %d positions, %d scalars, %d vectors"
                         % (n, len(S), len(V)))
    if any(len(v) != 3 for v in V):
        raise ValueError("se3T: type-1 features must be 3-vectors")
    t = float(temperature)
    if t <= 0.0:
        raise ValueError("se3T: the temperature must be positive")
    out_v, out_s, W = [], [], []
    for i in range(n):
        logits = []
        for j in range(n):
            g = invariant_features(P, i, j)
            logits.append((S[i] * S[j]
                           + radial_kernel(g["distance"], weights,
                                           sigma)) / t)
        m = max(logits)
        e = [math.exp(v - m) for v in logits]
        z = sum(e)
        w = [v / z for v in e]
        W.append(w)
        out_v.append([sum(w[j] * V[j][a] for j in range(n))
                      for a in range(3)])
        out_s.append(sum(w[j] * S[j] for j in range(n)))
    return {"type1": out_v, "type0": out_s, "weights": W,
            "note": "invariant weights, equivariant values -- an "
                    "invariant convex combination of equivariant "
                    "vectors is equivariant"}


def check_equivariance(positions, type0, type1, layer=None,
                       axis=(0.3, -0.7, 0.4), angle=1.1,
                       translation=(2.0, -1.0, 0.5), tol=1e-9):
    r"""Rotate and translate the input; the output must follow.

    The definition itself, checked to machine precision -- a model that
    only learned the symmetry cannot pass this.
    """
    f = layer if layer is not None else se3_attention
    P = [[float(v) for v in r] for r in k.mat(positions)]
    R = rotation_matrix(axis, angle)
    tv = [float(v) for v in k.vec(translation)]
    base = f(P, type0, type1)
    moved = f([[_apply(R, p)[a] + tv[a] for a in range(3)]
               for p in P], type0,
              [_apply(R, v) for v in k.mat(type1)])
    dev_v = 0.0
    for i in range(len(P)):
        want = _apply(R, base["type1"][i])
        got = moved["type1"][i]
        dev_v = max(dev_v, max(abs(want[a] - got[a])
                               for a in range(3)))
    dev_s = max(abs(base["type0"][i] - moved["type0"][i])
                for i in range(len(P)))
    dev_w = max(abs(base["weights"][i][j] - moved["weights"][i][j])
                for i in range(len(P)) for j in range(len(P)))
    return RichResult(payload={
        "estimate": dev_v, "type1_deviation": dev_v,
        "type0_deviation": dev_s, "weight_deviation": dev_w,
        "equivariant": dev_v < float(tol) and dev_s < float(tol),
        "weights_invariant": dev_w < float(tol),
        "method": "SE(3)-equivariance check; Fuchs et al. (2020)",
        "note": "type-1 outputs ROTATE with the input, type-0 outputs "
                "and the attention weights do not move at all",
    })


def cheatsheet():
    return ("se3T: a point cloud has no canonical orientation, so "
            "without a symmetry constraint the model must LEARN that a "
            "rotated molecule is the same molecule -- imperfectly, with "
            "no test-time guarantee. Equivariance builds it in: "
            "f(Rx+t) = R f(x) for type-1 outputs, f(x) for type-0. "
            "Mechanism: attention WEIGHTS are built only from "
            "INVARIANT quantities (distances, scalars), while the "
            "VALUES aggregated are equivariant -- an invariant convex "
            "combination of equivariant vectors is equivariant. The "
            "radial kernel sees the DISTANCE alone. Check it by "
            "rotating the input and comparing to machine precision.")


# compact alias per ledger/NAMING.md
se3transformer = se3_attention

# public names resolved by fn/_lazy_map.json
se3_transformer = se3_attention
