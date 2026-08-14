# morie.fn -- function file (rootcoder007/morie)
r"""NeRF: a scene as a continuous 5D function.

View synthesis is posed here as *optimising a representation* rather
than predicting pixels. A static scene is a continuous 5D function --
spatial location :math:`(x,y,z)` and viewing direction
:math:`(\theta,\phi)` -- returning volume density :math:`\sigma` and
view-dependent emitted radiance. A fully-connected, **non-convolutional**
network stores it: the weights *are* the scene.

**Density must not depend on direction, and the architecture enforces
it.** :math:`\sigma` is predicted from position alone; the direction is
injected only in the final layers, for colour. Otherwise the network
could fake a specularity by making geometry appear and disappear as
the camera moves, and would fit the training views while producing
nothing coherent in between.

**Rendering is classic volume rendering, and it is differentiable.**
Along a ray :math:`r(t) = o + td`,

.. math:: C(r) = \int_{t_n}^{t_f} T(t)\,\sigma(r(t))\,
          c(r(t), d)\, dt, \qquad
          T(t) = \exp\Big(-\int_{t_n}^{t}\sigma(r(s))ds\Big),

with :math:`T` the accumulated transmittance. Because the whole
operation is differentiable, the only input needed is a set of images
with known poses -- no 3D supervision at all.

**Positional encoding, or the result is blurry.** A plain MLP on raw
coordinates is biased toward low frequencies. Mapping inputs through
:math:`\gamma(p) = (\sin 2^0\pi p, \cos 2^0\pi p, \dots)` lets the
same network represent high-frequency detail, and the anchor shows the
encoding separating nearby points that raw coordinates leave nearly
identical.

**Hierarchical sampling** spends samples where density is: a coarse
network's weights along the ray become a PDF from which the fine
network draws.

References
----------
Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T.,
Ramamoorthi, R. & Ng, R. (2020) "NeRF: Representing Scenes as Neural
Radiance Fields for View Synthesis", *European Conference on Computer
Vision (ECCV 2020)*, LNCS 12346, 405-421,
doi:10.1007/978-3-030-58452-8_24, arXiv:2003.08934. The abstract and
Secs. 3-5: a scene represented by a fully-connected (non-convolutional)
deep network whose input is a single continuous 5D coordinate -- spatial
location and viewing direction -- and whose output is the volume
density and view-dependent emitted radiance at that location; views
synthesised by querying 5D coordinates along camera rays and using
classic volume rendering to project colours and densities into an
image; the observation that because volume rendering is naturally
differentiable the only required input is a set of images with known
camera poses; positional encoding; and hierarchical volume sampling.

Max, N. (1995) "Optical models for direct volume rendering", *IEEE
Transactions on Visualization and Computer Graphics* 1(2), 99-108,
doi:10.1109/2945.468400. The volume rendering integral.

Kerbl, B., Kopanas, G., Leimkuhler, T. & Drettakis, G. (2023) "3D
Gaussian Splatting for Real-Time Radiance Field Rendering", *ACM
Transactions on Graphics* 42(4), doi:10.1145/3592433. The explicit
alternative; implemented in :mod:`gsplat`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["positional_encoding", "volume_render", "sample_pdf",
           "ray_points", "density_is_view_independent"]

_EPS = 1e-12


def positional_encoding(p, L=10, include_input=True):
    r""":math:`\gamma(p) = (\sin 2^k\pi p, \cos 2^k\pi p)_{k<L}`.

    A plain MLP on raw coordinates is biased to low frequencies, so
    without this the reconstruction is blurry however long it trains.
    """
    v = [float(q) for q in k.vec(p)]
    if int(L) < 1:
        raise ValueError("nrfrad: L must be at least 1")
    out = list(v) if include_input else []
    for j in range(int(L)):
        f = (2.0 ** j) * math.pi
        for q in v:
            out.append(math.sin(f * q))
            out.append(math.cos(f * q))
    return out


def ray_points(origin, direction, t_near, t_far, n_samples,
               rng=None, stratified=True):
    r"""Sample points along a ray, stratified within equal bins.

    Stratification avoids the network only ever seeing the same fixed
    depths, which would let it overfit those and interpolate badly.
    """
    o = [float(v) for v in k.vec(origin)]
    d = [float(v) for v in k.vec(direction)]
    nrm = math.sqrt(sum(x * x for x in d))
    if nrm <= _EPS:
        raise ValueError("nrfrad: the ray direction is zero")
    d = [x / nrm for x in d]
    n = int(n_samples)
    if n < 1 or float(t_far) <= float(t_near):
        raise ValueError("nrfrad: need n >= 1 and t_far > t_near")
    step = (float(t_far) - float(t_near)) / n
    ts = []
    for i in range(n):
        lo = float(t_near) + i * step
        u = float(rng.uniform()) if (stratified and rng is not None) \
            else 0.5
        ts.append(lo + u * step)
    return {"t": ts, "points": [[o[a] + t * d[a] for a in range(3)]
                                for t in ts], "direction": d}


def volume_render(sigma, colour, t):
    r"""The rendering integral, discretised.

    :math:`T_i = \exp(-\sum_{j<i}\sigma_j\delta_j)`,
    :math:`\alpha_i = 1 - e^{-\sigma_i\delta_i}`,
    :math:`C = \sum_i T_i\alpha_i c_i` -- and the weights
    :math:`T_i\alpha_i` are what the hierarchical sampler reuses.
    """
    s = [float(v) for v in k.vec(sigma)]
    C = [[float(v) for v in r] for r in k.mat(colour)]
    ts = [float(v) for v in k.vec(t)]
    n = len(s)
    if not (len(C) == len(ts) == n):
        raise ValueError("nrfrad: sigma, colour and t differ in "
                         "length")
    if any(v < 0.0 for v in s):
        raise ValueError("nrfrad: density cannot be negative")
    deltas = [ts[i + 1] - ts[i] for i in range(n - 1)] + [1e10]
    T, acc, weights = 1.0, [0.0] * len(C[0]), []
    for i in range(n):
        a = 1.0 - math.exp(-s[i] * deltas[i])
        w = T * a
        weights.append(w)
        for c in range(len(acc)):
            acc[c] += w * C[i][c]
        T *= (1.0 - a)
    return {"colour": acc, "weights": weights,
            "accumulated_alpha": sum(weights),
            "transmittance_final": T,
            "note": "differentiable, which is why only posed IMAGES "
                    "are needed -- no 3D supervision"}


def sample_pdf(bins, weights, n_samples, rng, eps=1e-5):
    r"""Hierarchical sampling: draw where the coarse weights are.

    The coarse pass's rendering weights become a PDF, so the fine
    network's samples land near surfaces instead of in empty space.
    """
    b = [float(v) for v in k.vec(bins)]
    w = [float(v) + float(eps) for v in k.vec(weights)]
    if len(w) != len(b) - 1 and len(w) != len(b):
        raise ValueError("nrfrad: %d weights do not match %d bins"
                         % (len(w), len(b)))
    tot = sum(w)
    pdf = [v / tot for v in w]
    cdf, acc = [], 0.0
    for v in pdf:
        acc += v
        cdf.append(acc)
    out = []
    for _ in range(int(n_samples)):
        u = float(rng.uniform())
        i = 0
        while i < len(cdf) - 1 and u > cdf[i]:
            i += 1
        lo = b[i]
        hi = b[min(i + 1, len(b) - 1)]
        out.append(lo + (hi - lo) * float(rng.uniform()))
    return sorted(out)


def density_is_view_independent(model, point, directions, tol=1e-9):
    r"""Density must not change with the viewing direction.

    If it does, the network can fake specular highlights by making
    geometry appear and vanish with the camera -- fitting the training
    views and interpolating to nonsense.
    """
    p = [float(v) for v in k.vec(point)]
    ss = [float(model(p, [float(q) for q in k.vec(d)])["sigma"])
          for d in directions]
    dev = max(ss) - min(ss)
    return {"sigmas": ss, "max_deviation": dev,
            "view_independent": dev < float(tol),
            "note": "sigma from position alone; direction enters only "
                    "for colour"}


def cheatsheet():
    return ("nrfrad: a scene IS a continuous 5D function -- position "
            "plus viewing direction to density and radiance -- stored "
            "in a plain MLP; the weights are the scene. DENSITY must "
            "come from position ALONE (direction only affects colour), "
            "or the network fakes specularity by making geometry "
            "appear and vanish with the camera. Classic volume "
            "rendering, and because it is DIFFERENTIABLE the only "
            "input is posed images -- no 3D supervision. POSITIONAL "
            "ENCODING is not optional: a raw-coordinate MLP is "
            "low-frequency biased and renders blurry. Hierarchical "
            "sampling reuses the coarse weights as a PDF.")


# compact alias per ledger/NAMING.md
neuralradiancefield = volume_render

# public names resolved by fn/_lazy_map.json
nerf_radiance = volume_render
nerfradiance = volume_render
