"""Circulant embedding, power-law and random-phase fields, histogram transform,
coherent pairs and thin-plate splines.

Checked against R: both morie arms agree to 1e-13 (identical Philox draws);
thin_plate_spline equals fields::Tps to 1e-15 (tests/cross in R).
"""

import math

import pytest

from morie.fn._sci_core import kv
from morie.fn.rphase import random_phase_field
from morie.fn.sgsps import _corr, spectral_grf_sim
from morie.fn.spcfld import coherent_fields, histogram_transform, power_law_field
from morie.fn.thinpls import thin_plate_spline


def test_kv_matches_closed_form_half_order():
    for x in (0.001, 0.3, 5.0, 19.9, 20.1, 80.0):
        exact = math.sqrt(math.pi / (2 * x)) * math.exp(-x)
        assert abs(float(kv(0.5, x)) / exact - 1) < 1e-14


def test_embedding_reproduces_the_covariance_exactly():
    # the grid covariance implied by the embedding eigenvalues is the model covariance
    g = [(x * 0.5, y * 0.5) for x in range(5) for y in range(4)]
    r = spectral_grf_sim(g, "matern", {"range": 0.4, "nu": 1.5, "sill": 2.0}, seed=1)
    S = r.extra["eigenvalues"]
    Nx, Ny = r.extra["torus"]
    for i, j in ((0, 0), (1, 0), (2, 3), (4, 1)):
        c = sum(S[a][b] * math.cos(2 * math.pi * (a * i / Nx + b * j / Ny)) for a in range(Nx) for b in range(Ny))
        c /= Nx * Ny
        h = math.hypot(0.5 * i, 0.5 * j) / 0.4
        assert c == pytest.approx(2.0 * _corr(h, "matern", 1.5), abs=1e-12)
    assert r.extra["embedding_exact"]


def test_spectral_grf_is_deterministic_and_anisotropy_is_symmetric():
    g = [(x * 1.0, y * 1.0) for x in range(4) for y in range(4)]
    a = spectral_grf_sim(g, "gaussian", {"range": 2.0, "angle": 0.7, "ratio": 0.4}, n_sims=2, seed=3)
    b = spectral_grf_sim(g, "gaussian", {"range": 2.0, "angle": 0.7, "ratio": 0.4}, n_sims=2, seed=3)
    assert a.extra["simulations"].tolist() == b.extra["simulations"].tolist()
    S = a.extra["eigenvalues"]
    assert min(min(r) for r in S) > -1e-8 * max(max(r) for r in S)


def test_power_law_field_has_zero_mean_and_the_filter():
    r = power_law_field(8, 6, beta=1.7, dx=0.5, seed=3)
    assert abs(sum(sum(v) for v in r.field)) < 1e-12
    k = math.hypot(r.kx[1], r.ky[2])
    assert r.filter[1][2] == pytest.approx(k ** (-0.85), abs=1e-12)
    assert r.filter[0][0] == 0.0


def test_histogram_transform_is_rank_preserving():
    assert histogram_transform([0.3, -1.2, 0.8, 0.1], [10, 40, 20, 30]).transformed == [30.0, 10.0, 40.0, 20.0]
    out = histogram_transform([5.0, 1.0, 3.0], [0.0, 10.0, 20.0, 30.0, 40.0]).transformed
    assert out == [40.0, 0.0, 20.0]


def test_coherent_fields_mix_linearly():
    g = [(x * 1.0, y * 1.0) for x in range(3) for y in range(3)]
    one = coherent_fields(g, coherence=1.0, seed=2)
    assert one.first == one.second
    z2 = spectral_grf_sim(g, seed=3).extra["simulations"].tolist()[0]
    r = coherent_fields(g, coherence=0.6, seed=2)
    assert r.second == pytest.approx([0.6 * a + 0.8 * b for a, b in zip(r.first, z2)], abs=1e-12)


def test_thin_plate_spline_interpolates_and_reproduces_planes():
    pts = [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5), (0.2, 0.9)]
    y = [1 + 2 * a - 3 * b for a, b in pts]
    r = thin_plate_spline(pts, y, lam=0.5, newdata=[(0.3, 0.3)])
    assert r.fitted == pytest.approx(y, abs=1e-12)  # a plane has zero bending energy
    assert r.predicted[0] == pytest.approx(1 + 0.6 - 0.9, abs=1e-12)
    z = [0.0, 1.0, 1.0, 2.0, 1.5, 0.4]
    s = thin_plate_spline(pts, z)
    assert s.fitted == pytest.approx(z, abs=1e-12) and s.df == 6.0
    assert [round(v, 6) for v in thin_plate_spline(pts[:5], z[:5], newdata=[(0.5, 0.5), (0.25, 0.75)]).predicted] == [
        1.5,
        1.294285,
    ]


def test_random_phase_field_documented_and_spectral_radius():
    r = random_phase_field([(0.0, 0.0), (1.0, 0.5)], "gaussian", n_waves=4, seed=2)
    assert [round(v, 6) for v in r.field] == [0.481186, -0.249627]
    big = random_phase_field([(0.0, 0.0)], "exponential", range_=0.8, n_waves=20000, seed=6)
    emp = sum(math.cos(o[0] * 1.0) for o in big.omega) / 20000
    assert abs(emp - _corr(1.0 / 0.8, "exponential", 0.5)) < 4 * math.sqrt(0.5 / 20000)
