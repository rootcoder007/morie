"""vgextra: estimators by brute force, anisotropy geometry, envelope/jackknife recomputation, fractal slopes."""

import itertools
import math

import pytest

from morie.fn.vgextra import (
    anisotropic_lag,
    sample_variogram_nd,
    variogram_cloud_box,
    variogram_envelope,
    variogram_fractal,
    variogram_jackknife,
    windowed_semivariance,
    zonal_semivariance,
)
from morie.fn.vgmods import vgm_semivariance

P3 = [(0.0, 0.0, 0.0), (1.0, 0.2, 0.0), (0.3, 1.1, 0.5), (1.4, 1.2, 0.9), (0.5, 0.4, 1.6), (2.0, 0.1, 1.1)]
Z = [1.0, 2.5, 1.8, 3.9, 2.2, 4.4]
Z2 = [0.3, 0.9, 0.2, 1.4, 1.0, 1.8]


def brute(est, B, z2=None, direction=None, tol=90.0):
    out = []
    for lo, hi in zip(B, B[1:]):
        vals = []
        for i, j in itertools.combinations(range(6), 2):
            h = [b - a for a, b in zip(P3[i], P3[j])]
            d = math.sqrt(sum(v * v for v in h))
            if direction is not None:
                u = [v / math.sqrt(sum(w * w for w in direction)) for v in direction]
                if abs(sum(a * b for a, b in zip(h, u))) / d < math.cos(math.radians(tol)) - 1e-12:
                    continue
            if lo < d <= hi:
                dz = Z[i] - Z[j]
                vals.append(
                    {
                        "classical": dz * dz,
                        "madogram": abs(dz),
                        "rodogram": abs(dz) ** 0.5,
                        "cross": dz * ((z2 or Z2)[i] - (z2 or Z2)[j]),
                    }[est]
                )
        if vals:
            out.append(sum(vals) / (2 * len(vals)))
    return out


def test_estimators_and_directions():
    B = [0.0, 1.0, 1.8, 3.0]
    for est in ("classical", "madogram", "rodogram", "cross"):
        r = sample_variogram_nd(Z, P3, B, estimator=est, z2=Z2)
        assert r.gamma == pytest.approx(brute(est, B), abs=1e-14)
    d = sample_variogram_nd(Z, P3, B, direction=(0, 0, 1), tol=45.0)
    assert d.gamma == pytest.approx(brute("classical", B, direction=(0, 0, 1), tol=45.0), abs=1e-14)
    with pytest.raises(ValueError):
        sample_variogram_nd(Z, P3, B, estimator="bogus")


def test_anisotropy():
    assert anisotropic_lag((0.0, 2.0, 0.0), azimuth=0.0, ratio1=0.5) == pytest.approx(2.0)
    assert anisotropic_lag((2.0, 0.0, 0.0), azimuth=90.0, ratio1=0.25) == pytest.approx(2.0)  # major axis east
    assert anisotropic_lag((0.0, 0.0, 1.0), ratio2=0.1) == pytest.approx(10.0)
    # rotations preserve length when all ratios are 1
    assert anisotropic_lag((1.0, 2.0, 2.0), azimuth=33.0, dip=20.0, rake=10.0) == pytest.approx(3.0)
    m = {"model": "Sph", "psill": 2.0, "range": 3.0}
    z = {"model": "Exp", "psill": 0.5, "range": 1.0}
    h = (1.0, 1.0, 0.5)
    assert zonal_semivariance(h, m, z) == pytest.approx(vgm_semivariance(1.5, m) + vgm_semivariance(0.5, z))


def test_envelope_jackknife_cloud_fractal_window():
    B = [0.0, 1.2, 2.5]
    env = variogram_envelope(Z, P3, B, nsim=39, seed=3)
    assert all(lo <= hi for lo, hi in zip(env.lower, env.upper))
    assert env.outside == [not (lo <= g <= hi) for g, lo, hi in zip(env.gamma, env.lower, env.upper)]
    jk = variogram_jackknife(Z, P3, B)
    reps = [sample_variogram_nd(Z[:i] + Z[i + 1 :], P3[:i] + P3[i + 1 :], B).gamma for i in range(6)]
    m = [sum(r[b] for r in reps) / 6 for b in range(2)]
    assert jk.se == pytest.approx([math.sqrt(5 / 6 * sum((r[b] - m[b]) ** 2 for r in reps)) for b in range(2)])
    cb = variogram_cloud_box(Z, P3, B)
    assert cb.n == [
        sum(1 for i, j in itertools.combinations(range(6), 2) if 0 < math.dist(P3[i], P3[j]) <= 1.2),
        sum(1 for i, j in itertools.combinations(range(6), 2) if 1.2 < math.dist(P3[i], P3[j]) <= 2.5),
    ]
    fr = variogram_fractal([1, 2, 4, 8], [3 * h**1.4 for h in (1, 2, 4, 8)], dim=2)
    assert (fr.hurst, fr.fractal_dimension) == pytest.approx((0.7, 2.3))
    ws = windowed_semivariance([0.0, 2.0, 1.0, 3.0, 6.0], [1, 2], window=3)
    flat = [v for row in ws.gamma for v in row]
    assert flat == pytest.approx([(4 + 1) / 4, 1 / 2, (1 + 4) / 4, 1 / 2, (4 + 9) / 4, 25 / 2])
