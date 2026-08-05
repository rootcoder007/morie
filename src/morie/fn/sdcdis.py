# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Geomasking: spatial data distortion for privacy protection."""

from __future__ import annotations

import math

from . import _array_core as np
from . import _tail1core as C
from ._richresult import RichResult

__all__ = ["spatial_data_distortion"]


def spatial_data_distortion(coords, noise_radius, seed=1):
    r"""Displace point locations by bounded random noise (geomasking) and
    report how much the spatial signal moved.

    Each point is shifted to a uniformly random location inside a disc of
    radius :math:`r` centred on it.  Uniformity over the disc requires the
    radial draw to be

    .. math::  \rho = r\sqrt{u_1}, \qquad \theta = 2\pi u_2,

    with :math:`u_1, u_2` independent uniforms -- taking
    :math:`\rho = r u_1` would over-concentrate points near the centre,
    which is exactly the failure mode that makes a mask look stronger than
    it is.  The displaced point is
    :math:`(x + \rho\cos\theta,\ y + \rho\sin\theta)`.

    Under this mask the expected displacement is
    :math:`E[\rho] = \tfrac{2}{3} r` and
    :math:`E[\rho^2] = \tfrac{1}{2} r^2`, so both the mean and the RMS
    displacement have closed forms the sample values can be checked
    against; they are returned as ``expected_displacement`` and
    ``expected_rms``.

    The mask preserves the mean centre in expectation but not exactly; the
    realised shift of the mean centre is reported, as is the change in mean
    pairwise distance, the quantity most spatial analyses actually depend
    on.  Allshouse et al. evaluate exactly this trade-off between
    re-identification risk and analytic distortion.

    The uniform stream is the Lehmer minstd generator shared with every
    other arm of this package, so a given ``seed`` reproduces the same mask
    in Python and in R bit for bit.

    Parameters
    ----------
    coords : array-like
        Point coordinates, n rows by 2 columns (x, y).
    noise_radius : float
        Maximum displacement r, non-negative.
    seed : int, default 1
        Seed for the shared minstd stream.

    Returns
    -------
    RichResult
        ``estimate`` is the mean realised displacement.  ``masked`` holds
        the displaced coordinates.

    References
    ----------
    Allshouse, W. B., Fitch, M. K., Hampton, K. H., Gesink, D. C.,
    Doherty, I. A., Leone, P. A., Serre, M. L. & Miller, W. C. (2010).
    Geomasking sensitive health data and privacy protection: an evaluation
    using an E911 database. Geocarto International 25(6), 443-452.
    doi:10.1080/10106049.2010.496496
    """
    P = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(coords, dtype=float)).tolist()]
    n = len(P)
    if n == 0:
        raise ValueError("spatial_data_distortion: coords is empty")
    if any(len(r) != 2 for r in P):
        raise ValueError("spatial_data_distortion: coords must have exactly two columns")
    r = float(noise_radius)
    if r < 0.0:
        raise ValueError("spatial_data_distortion: noise_radius must be non-negative")

    rng = C.Lcg(seed)
    masked = []
    disp = []
    for k in range(n):
        rho = r * math.sqrt(rng.unif())
        th = 2.0 * math.pi * rng.unif()
        masked.append([P[k][0] + rho * math.cos(th), P[k][1] + rho * math.sin(th)])
        disp.append(rho)

    def centre(Q):
        return [sum(q[0] for q in Q) / n, sum(q[1] for q in Q) / n]

    c0 = centre(P)
    c1 = centre(masked)
    centre_shift = math.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2)

    def mean_pair(Q):
        if n < 2:
            return float("nan")
        s = 0.0
        m = 0
        for a in range(n):
            for b in range(a + 1, n):
                s += math.sqrt((Q[a][0] - Q[b][0]) ** 2 + (Q[a][1] - Q[b][1]) ** 2)
                m += 1
        return s / m

    mp0 = mean_pair(P)
    mp1 = mean_pair(masked)

    mean_disp = sum(disp) / n
    rms = math.sqrt(sum(d * d for d in disp) / n)

    return RichResult(
        payload={
            "estimate": mean_disp,
            "masked": masked,
            "displacement": disp,
            "mean_displacement": mean_disp,
            "max_displacement": max(disp),
            "rms_displacement": rms,
            "expected_displacement": 2.0 * r / 3.0,
            "expected_rms": r / math.sqrt(2.0),
            "centre_shift": centre_shift,
            "mean_pairwise_before": mp0,
            "mean_pairwise_after": mp1,
            "mean_pairwise_change": mp1 - mp0,
            "noise_radius": r,
            "n": float(n),
            "seed": float(int(seed)),
            "method": "Geomasking by uniform displacement within a disc (Allshouse et al. 2010)",
        }
    )


def cheatsheet():
    return "sdcdis: geomasking, uniform-disc spatial data distortion"


# compact alias per ledger/NAMING.md
spatialdatadistortion = spatial_data_distortion
