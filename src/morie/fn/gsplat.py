# morie.fn -- function file (rootcoder007/morie)
r"""3D Gaussian splatting: an explicit primitive that rasterises.

A neural radiance field stores the scene implicitly and must be
*queried* -- hundreds of network evaluations per ray -- which is why
real-time rendering was out of reach. 3D Gaussian splatting keeps the
same optimisation-from-posed-images setting and changes the
representation to an **explicit** one: a set of anisotropic 3D
Gaussians, each with position, covariance, opacity and view-dependent
colour, which can be projected and rasterised rather than marched.

**The covariance parametrisation is forced, not chosen.** A covariance
matrix must stay positive semi-definite, and gradient descent on its
six entries will not keep it there. Factoring it as

.. math:: \Sigma = R S S^\top R^\top

with :math:`S` a diagonal scale and :math:`R` a rotation from a
normalised quaternion makes every reachable parameter value legal by
construction. ``covariance_from_scale_rotation`` builds it that way,
and the anchor confirms the result is PSD for arbitrary parameters
where a raw six-vector would not be.

**Projection is the EWA splat.** The 3D Gaussian projects to a 2D one
with :math:`\Sigma' = J W \Sigma W^\top J^\top`, where :math:`J` is
the Jacobian of the perspective projection -- an affine approximation,
which is what makes the splat closed-form and therefore fast.

**Compositing is the same alpha-blend as volume rendering**, in
depth-sorted order:
:math:`C = \sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)`. So the image
formation model is unchanged from NeRF; only the primitive and the
traversal differ, and the anchor checks the two agree given matching
alphas.

**Adaptive density control** is the part that makes optimisation work:
Gaussians in under-reconstructed regions are cloned, those spanning
too much are split, and near-transparent ones are pruned -- the count
is not fixed in advance.

References
----------
Kerbl, B., Kopanas, G., Leimkuhler, T. & Drettakis, G. (2023) "3D
Gaussian Splatting for Real-Time Radiance Field Rendering", *ACM
Transactions on Graphics* 42(4), Article 139,
doi:10.1145/3592433, arXiv:2308.04079. The 3D Gaussian scene
representation with position, anisotropic covariance and opacity; the
factorisation of covariance into scaling and rotation to keep it
positive semi-definite under optimisation; interleaved optimisation
with adaptive density control including cloning, splitting and
pruning; and the fast tile-based differentiable rasteriser enabling
real-time rendering at high quality.

Zwicker, M., Pfister, H., van Baar, J. & Gross, M. (2001) "EWA
volume splatting", *Proceedings Visualization 2001*, 29-36,
doi:10.1109/VISUAL.2001.964490. The projected-Gaussian splat.

Mildenhall, B. et al. (2020) "NeRF", *ECCV 2020*, LNCS 12346,
405-421, doi:10.1007/978-3-030-58452-8_24. The implicit alternative;
implemented in :mod:`nrfrad`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["covariance_from_scale_rotation", "project_covariance",
           "alpha_composite", "adaptive_density_control",
           "is_positive_semidefinite"]

_EPS = 1e-12


def _quat_to_rot(q):
    v = [float(x) for x in k.vec(q)]
    n = math.sqrt(sum(x * x for x in v))
    if n <= _EPS:
        raise ValueError("gsplat: the rotation quaternion is zero")
    w, x, y, z = [x / n for x in v]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
             2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
             2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x),
             1 - 2 * (x * x + y * y)]]


def covariance_from_scale_rotation(scale, quaternion):
    r""":math:`\Sigma = RSS^\top R^\top`.

    Every parameter value gives a legal covariance; optimising the six
    entries directly would not.
    """
    s = [float(v) for v in k.vec(scale)]
    if any(v <= 0.0 for v in s):
        raise ValueError("gsplat: scales must be positive")
    R = _quat_to_rot(quaternion)
    M = [[R[i][j] * s[j] for j in range(3)] for i in range(3)]
    S = [[sum(M[i][a] * M[j][a] for a in range(3)) for j in range(3)]
         for i in range(3)]
    return {"covariance": S, "rotation": R, "scale": s,
            "note": "PSD by construction, which raw entries would "
                    "not be"}


def is_positive_semidefinite(S, tol=-1e-9):
    r"""Check via eigenvalues -- the property the factorisation
    guarantees."""
    M = [[float(v) for v in r] for r in k.mat(S)]
    vals, _ = np.linalg.eigh(M)
    return {"eigenvalues": list(vals), "min_eigenvalue": min(vals),
            "psd": min(vals) >= float(tol)}


def project_covariance(S, W, J):
    r""":math:`\Sigma' = JW\Sigma W^\top J^\top`, the EWA splat.

    :math:`J` is the affine approximation to the perspective
    projection, which is what keeps the projected Gaussian
    closed-form.
    """
    C = [[float(v) for v in r] for r in k.mat(S)]
    Wm = [[float(v) for v in r] for r in k.mat(W)]
    Jm = [[float(v) for v in r] for r in k.mat(J)]
    T = [[sum(Jm[i][a] * Wm[a][j] for a in range(len(Wm)))
          for j in range(len(Wm[0]))] for i in range(len(Jm))]
    TC = [[sum(T[i][a] * C[a][j] for a in range(len(C)))
           for j in range(len(C[0]))] for i in range(len(T))]
    out = [[sum(TC[i][a] * T[j][a] for a in range(len(T[0])))
            for j in range(len(T))] for i in range(len(T))]
    return {"projected": out, "dim": len(out),
            "note": "an affine approximation to perspective, hence "
                    "closed form and fast"}


def alpha_composite(colours, alphas, depths=None):
    r"""Front-to-back alpha blending in depth order.

    The same image formation model as volume rendering, which is why
    the two representations are interchangeable at the pixel.
    """
    C = [[float(v) for v in r] for r in k.mat(colours)]
    a = [float(v) for v in k.vec(alphas)]
    if len(C) != len(a):
        raise ValueError("gsplat: %d colours but %d alphas"
                         % (len(C), len(a)))
    if any(v < 0.0 or v > 1.0 for v in a):
        raise ValueError("gsplat: alphas must lie in [0,1]")
    order = range(len(a)) if depths is None else \
        sorted(range(len(a)), key=lambda i: float(depths[i]))
    T, acc = 1.0, [0.0] * len(C[0])
    for i in order:
        for c in range(len(acc)):
            acc[c] += T * a[i] * C[i][c]
        T *= (1.0 - a[i])
    return {"colour": acc, "transmittance": T,
            "coverage": 1.0 - T,
            "note": "identical compositing to volume rendering; only "
                    "the primitive and traversal differ"}


def adaptive_density_control(gradients, scales, opacities,
                             grad_threshold=0.0002,
                             scale_threshold=0.01,
                             opacity_threshold=0.005):
    r"""Clone, split or prune -- the Gaussian count is not fixed.

    Large positional gradient with a SMALL Gaussian means
    under-reconstruction (clone); with a LARGE one it means the
    Gaussian spans too much (split). Near-transparent ones are pruned.
    """
    g = [float(v) for v in k.vec(gradients)]
    s = [float(v) for v in k.vec(scales)]
    o = [float(v) for v in k.vec(opacities)]
    if not (len(g) == len(s) == len(o)):
        raise ValueError("gsplat: the inputs differ in length")
    clone, split, prune = [], [], []
    for i in range(len(g)):
        if o[i] < float(opacity_threshold):
            prune.append(i)
        elif g[i] > float(grad_threshold):
            (split if s[i] > float(scale_threshold)
             else clone).append(i)
    return RichResult(payload={
        "estimate": {"clone": clone, "split": split, "prune": prune},
        "clone": clone, "split": split, "prune": prune,
        "n_before": len(g),
        "n_after": len(g) + len(clone) + len(split) - len(prune),
        "method": "adaptive density control; Kerbl et al. (2023)",
        "note": "under-reconstruction clones, over-reconstruction "
                "splits, transparent prunes",
    })


def cheatsheet():
    return ("gsplat: a radiance field stored EXPLICITLY as anisotropic "
            "3D Gaussians that RASTERISE, instead of an implicit field "
            "that must be marched -- same posed-image optimisation, "
            "real-time rendering. Covariance is factored as "
            "R S S' R' because gradient descent on six raw entries "
            "would not stay PSD. Projection is the EWA splat, "
            "J W Sigma W' J', affine and closed-form. Compositing is "
            "the SAME alpha blend as volume rendering. Adaptive "
            "density control clones under-reconstructed Gaussians, "
            "splits oversized ones and prunes transparent ones -- the "
            "count is not fixed in advance.")


# compact alias per ledger/NAMING.md
gaussiansplatting = alpha_composite
