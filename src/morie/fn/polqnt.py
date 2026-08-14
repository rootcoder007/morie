# morie.fn -- function file (rootcoder007/morie)
"""PolarQuant polar-transformation quantization (uniform-bin variant)."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["polqnt", "polarquant"]

# NOTE ON ATTRIBUTION: the generated stub credited "PolarQuant
# (Tang 2024)".  No such paper exists; PolarQuant is Han, Kacham,
# Karbasi, Mirrokni and Zandieh (2025), arXiv:2502.02617.  Implemented
# from that paper.


def _polar_decompose(x):
    # Definition 1: level-1 angles atan2 in [0, 2pi) over coordinate
    # pairs; higher-level angles atan of radius ratios in [0, pi/2].
    levels = []
    r = list(x)
    while len(r) > 1:
        ang = []
        nxt = []
        first = not levels
        for j in range(0, len(r), 2):
            a, b = r[j], r[j + 1]
            rad = math.hypot(a, b)
            if first:
                t = math.atan2(b, a)
                if t < 0.0:
                    t += 2.0 * math.pi
            else:
                t = math.atan2(b, a)  # a, b >= 0 so t in [0, pi/2]
            ang.append(t)
            nxt.append(rad)
        levels.append(ang)
        r = nxt
    return levels, r[0]


def _polar_reconstruct(levels, radius):
    r = [radius]
    for ell in range(len(levels) - 1, -1, -1):
        ang = levels[ell]
        nxt = []
        for j, t in enumerate(ang):
            nxt.append(r[j] * math.cos(t))
            nxt.append(r[j] * math.sin(t))
        r = nxt
    return r


def polarquant(x, bits_first=4, bits_rest=2, quantize=True):
    r"""Polar-transformation compression of a vector (Han et al. 2025).

    Cartesian-to-polar transformation exactly as PolarQuant
    Definition 1: for d a power of two the vector becomes
    :math:`\log_2 d` levels of angles (level 1 in :math:`[0, 2\pi)`,
    higher levels in :math:`[0, \pi/2]`) plus one radius; the radius is
    kept in full precision and only angles are quantized, at
    ``bits_first`` bits for level 1 and ``bits_rest`` bits for the rest
    (the 4/2 split is the configuration of the paper, section 5).

    Quantizer per angle range :math:`[0, W)`: :math:`2^b` uniform
    intervals with midpoint reconstruction,
    :math:`q = \min(\lfloor \psi 2^b / W \rfloor, 2^b - 1)`,
    :math:`\hat\psi = (q + 1/2) W / 2^b`.  The paper computes
    density-optimal centroids for the sinusoidal angle densities (its
    section 4); this implementation ships the uniform-interval midpoint
    variant and says so -- the transform, storage layout and
    reconstruction are the paper, the centroid rule is the uniform
    simplification.

    With ``quantize=False`` the reconstruction is exact (the transform
    is a bijection), which is the test-suite roundtrip anchor.

    Parameters
    ----------
    x : array-like
        Input vector; length must be a power of two, at least 2.
    bits_first : int
        Bits for level-1 angles (range [0, 2pi)).
    bits_rest : int
        Bits for higher-level angles (range [0, pi/2]).
    quantize : bool
        If False, skip quantization (exact reconstruction).

    Returns
    -------
    RichResult
        ``reconstruction``, ``radius``, ``codes`` (per-level index
        lists, flattened level by level), ``mse``, ``relative_l2``
        (reconstruction error over the input norm), ``bits_per_coord``.

    References
    ----------
    Han, I., Kacham, P., Karbasi, A., Mirrokni, V., & Zandieh, A.
        (2025). PolarQuant: quantizing KV caches with polar
        transformation. arXiv:2502.02617. Definition 1 (transformation),
        sections 4-5 (quantization and bit budget).
        Local source: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/han-etal-2025-polarquant-kv-polar-transformation-arxiv2502.02617.pdf
    """
    v = [float(t) for t in x]
    d = len(v)
    if d < 2 or (d & (d - 1)) != 0:
        raise ValueError("length of x must be a power of two, at least 2")
    b1 = int(bits_first)
    br = int(bits_rest)
    if b1 < 1 or br < 1:
        raise ValueError("bit widths must be at least 1")
    levels, radius = _polar_decompose(v)
    codes = []
    if quantize:
        qlevels = []
        for ell, ang in enumerate(levels):
            W = 2.0 * math.pi if ell == 0 else 0.5 * math.pi
            nb = b1 if ell == 0 else br
            K = 1 << nb
            qa = []
            for t in ang:
                q = int(t * K / W)
                if q > K - 1:
                    q = K - 1
                if q < 0:
                    q = 0
                codes.append(float(q))
                qa.append((q + 0.5) * W / K)
            qlevels.append(qa)
        rec = _polar_reconstruct(qlevels, radius)
    else:
        rec = _polar_reconstruct(levels, radius)
    err2 = sum((rec[i] - v[i]) ** 2 for i in range(d))
    nrm2 = sum(t * t for t in v)
    nlev = len(levels)
    nbits = (d // 2) * b1 + sum((d >> (ell + 1)) * br
                                for ell in range(1, nlev))
    return RichResult(payload={
        "reconstruction": rec, "estimate": rec, "radius": radius,
        "codes": codes, "mse": err2 / d,
        "relative_l2": math.sqrt(err2 / nrm2) if nrm2 > 0 else 0.0,
        "bits_per_coord": nbits / d, "n": float(d),
        "method": "PolarQuant polar-transformation quantization"})


#: Primary name for the module.
polqnt = polarquant


def cheatsheet():
    return "polqnt: PolarQuant polar-transformation quantization (Han et al. 2025)."

# public names resolved by fn/_lazy_map.json
polar_quantization = polarquant
