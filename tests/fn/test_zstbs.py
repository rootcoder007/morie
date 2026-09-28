"""Tests for zstbs.turning_bands (turning bands simulation)."""

import math

import pytest

from morie.fn.zstbs import turning_bands, turningbands


def test_origin_field_is_the_phase_sum():
    # at x = 0 every band contributes sqrt(2/M) sum_m cos(phi_m): the field is model-free there
    from morie.fn._rng import random_uniform

    L, M = 3, 4
    r = turning_bands([(0.0, 0.0)], "exponential", n_bands=L, n_waves=M, seed=5)
    want = 0.0
    for k in range(L):
        up = random_uniform(M, seed=5, stream=3 * k + 2)
        want += math.sqrt(2.0 / M) * sum(math.cos(2.0 * math.pi * float(v)) for v in up)
    assert r.field[0] == pytest.approx(want / math.sqrt(L), abs=1e-12)


def test_single_band_single_wave_formula():
    from morie.fn._rng import random_uniform

    x = (0.7, -0.2)
    r = turning_bands([x], "gaussian", n_bands=1, n_waves=1, seed=9, range_=2.0, sill=3.0)
    off = float(random_uniform(1, seed=9, stream=0)[0])
    u = (math.cos(math.pi * off), math.sin(math.pi * off))
    rad = 2.0 / 2.0 * math.sqrt(-math.log(float(random_uniform(1, seed=9, stream=1)[0])))
    ph = 2.0 * math.pi * float(random_uniform(1, seed=9, stream=2)[0])
    want = math.sqrt(3.0) * math.sqrt(2.0) * math.cos(rad * (x[0] * u[0] + x[1] * u[1]) + ph)
    assert r.field[0] == pytest.approx(want, abs=1e-12)


def test_docstring_example():
    r = turning_bands([(0.0, 0.0), (1.0, 0.5)], "gaussian", n_bands=4, n_waves=3, seed=2)
    assert [round(v, 6) for v in r.field] == [0.881358, 0.400421]


def test_3d_directions_are_unit_and_rotated():
    r = turning_bands([(0.0, 0.0, 0.0)], "exponential", n_bands=10, n_waves=2, seed=4)
    for d in r.directions:
        assert math.hypot(*d) == pytest.approx(1.0, abs=1e-12)
    # the random rotation moves the lattice off its unrotated first point (0, 0, 1 - 1/(2L))
    assert abs(r.directions[0][2] - (1.0 - 0.05)) > 1e-6


def test_3d_exponential_radial_quantile_inverts_cdf():
    from morie.fn.zstbs import _radial_3d_exp

    for u in (0.1, 0.5, 0.93):
        r = _radial_3d_exp(u, 1.0)
        assert (2.0 / math.pi) * (math.atan(r) - r / (1.0 + r * r)) == pytest.approx(u, abs=1e-12)


@pytest.mark.parametrize(
    "pts,model,kw,rho",
    [
        ([(0, 0), (0.8, 0)], "exponential", {}, math.exp(-0.8)),
        ([(0, 0), (0.8, 0)], "gaussian", {}, math.exp(-0.64)),
        ([(0, 0), (0.8, 0)], "matern", {"nu": 1.5}, 1.8 * math.exp(-0.8)),
        ([(0, 0), (0, 0.4)], "exponential", {"ratio": 0.5}, math.exp(-0.8)),
        ([(0, 0, 0), (0, 0.8, 0)], "exponential", {}, math.exp(-0.8)),
        ([(0, 0, 0), (0.8, 0, 0)], "gaussian", {}, math.exp(-0.64)),
    ],
)
def test_monte_carlo_covariance(pts, model, kw, rho):
    # E[Z(0) Z(h)] = rho(|h|): the replicate mean is within 4 standard errors
    R = 1500
    prods = [turning_bands(pts, model, n_bands=8, n_waves=8, seed=20000 + s, **kw).field for s in range(R)]
    p = [f[0] * f[1] for f in prods]
    m = sum(p) / R
    se = math.sqrt(sum((v - m) ** 2 for v in p) / (R - 1) / R)
    assert abs(m - rho) < 4.0 * se


def test_validation_and_alias():
    assert turningbands is turning_bands
    with pytest.raises(ValueError):
        turning_bands([(0.0, 0.0, 0.0)], "matern")
    with pytest.raises(ValueError):
        turning_bands([(0.0,)], "gaussian")
    with pytest.raises(ValueError):
        turning_bands([(0.0, 0.0)], "gaussian", ratio=0.0)
