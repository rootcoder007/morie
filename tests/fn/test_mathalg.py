"""mathalg: algorithms checked by their defining properties."""

import math

import pytest

from morie.fn._rng import random_normal
from morie.fn.mathalg import (
    legendre_polynomials,
    logit_proportion,
    mc_standard_error,
    panter_dite_bound,
    polar_transform,
    pollard_rho,
    resolution_refutation,
)


def _probably_prime(p):
    return p > 1 and all(pow(a, p - 1, p) == 1 for a in (2, 3, 5, 7, 11) if a % p)


def test_pollard_factorisations():
    for n in (8051, 10403, 600851475143, 2**40 - 87, 3 * 5 * 7 * 1000003):
        f = pollard_rho(n)
        assert math.prod(f) == n and all(_probably_prime(p) for p in f)
    with pytest.raises(ValueError):
        pollard_rho(2**53)


def test_legendre_orthogonality():
    # Gauss-Legendre with 6 nodes integrates degree <= 11 exactly
    nodes = [-0.9324695142, -0.6612093865, -0.2386191861, 0.2386191861, 0.6612093865, 0.9324695142]
    w = [0.1713244924, 0.3607615730, 0.4679139346, 0.4679139346, 0.3607615730, 0.1713244924]
    P = legendre_polynomials(nodes, 4, normalized=True)
    for a in range(5):
        for b in range(5):
            s = sum(wi * p[a] * p[b] for wi, p in zip(w, P))
            assert s == pytest.approx(1.0 if a == b else 0.0, abs=1e-8)
    assert legendre_polynomials([1.0], 7)[0] == [1.0] * 8


def test_resolution():
    unsat = resolution_refutation([[1], [-1, 2], [-2]])
    assert unsat.unsatisfiable
    sat = resolution_refutation([[1, 2], [-1, 2], [3]])
    assert not sat.unsatisfiable and [] not in sat.clauses


def test_polar_transform_roundtrip_and_norm():
    z = [float(v) for v in random_normal(32, seed=7)]
    p = polar_transform(z)
    assert p.radius == pytest.approx(math.sqrt(sum(v * v for v in z)), rel=1e-14)
    assert len(p.angles) == 31
    assert polar_transform((p.radius, p.angles), inverse=True) == pytest.approx(z, abs=1e-12)
    with pytest.raises(ValueError):
        polar_transform([1.0, 2.0, 3.0])


def test_quantisation_proportions_and_mcse():
    r = panter_dite_bound(5, 4.0)
    assert r.mse == pytest.approx(math.sqrt(3) * math.pi / 2 * 4.0 / 1024, rel=1e-15)
    lp = logit_proportion([6], [24])
    assert lp.yi[0] == pytest.approx(math.log(0.25 / 0.75), rel=1e-15)
    assert lp.vi[0] == pytest.approx(1 / (24 * 0.25) + 1 / (24 * 0.75), rel=1e-14)
    iid = [float(v) for v in random_normal(4000, seed=8)]
    m = mc_standard_error(iid)
    sd = math.sqrt(sum((v - sum(iid) / 4000) ** 2 for v in iid) / 4000)
    assert m.mcse == pytest.approx(sd / math.sqrt(m.ess), rel=1e-12)
    assert 0.5 < m.ess / 4000 < 2.0
