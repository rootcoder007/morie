"""Tests for analytic: Laplace transforms, Laurent coefficients, Walsh-Hadamard, polynomials."""

import cmath
import math

from morie.fn.analytic import (
    functional_norm,
    hadamard_inverse,
    hadamard_transform,
    heat_equation_series,
    laplace_transform_num,
    laurent_coefficients,
    poly_expand,
    poly_factor_mod_p,
    quadratic_roots,
    rational_cancel,
    stehfest_inverse,
    talbot_inverse,
)


def test_laplace_transform_closed_forms():
    s = [0.3, 1.0, 2.5]
    got = laplace_transform_num(lambda t: t * math.exp(-0.5 * t) + math.sin(t), s)
    for g, v in zip(got, s):
        assert abs(g - (1 / (v + 0.5) ** 2 + 1 / (v * v + 1))) <= 1e-10


def test_talbot_and_stehfest_invert_known_transforms():
    for t in (0.2, 1.0, 3.7):
        want = math.sin(t) + t * math.exp(-2 * t)
        assert abs(talbot_inverse(lambda s: 1 / (s * s + 1) + 1 / (s + 2) ** 2, [t])[0] - want) <= 1e-10
        assert abs(stehfest_inverse(lambda s: 1 / (s * (s + 1)), [t])[0] - (1 - math.exp(-t))) <= 1e-5


def test_laurent_coefficients_of_exp_over_z_squared():
    r = laurent_coefficients(lambda z: cmath.exp(z) / (z * z), 0, [-2, -1, 0, 1, 3], n=64)
    for k, re, im in zip(r.orders, r.real, r.imag):
        assert abs(re - 1 / math.factorial(k + 2)) <= 1e-14 and abs(im) <= 1e-14


def test_hadamard_matches_matrix_and_is_involution():
    x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    y = hadamard_transform(x)
    for i in range(8):
        want = sum((-1) ** bin(i & j).count("1") * x[j] for j in range(8)) / math.sqrt(8)
        assert abs(y[i] - want) <= 1e-12
    assert max(abs(a - b) for a, b in zip(hadamard_inverse(y), x)) <= 1e-12


def test_poly_expand_and_rational_cancel():
    assert poly_expand([[1, 1]], [3]) == [1.0, 3.0, 3.0, 1.0]
    r = rational_cancel(
        poly_expand([[1, 1], [2, 3], [-5, 0, 7]], [2, 1, 1]), poly_expand([[1, 1], [4, 0, 2], [-5, 0, 7]], [1, 1, 2])
    )
    assert r.numerator == [2.0, 5.0, 3.0]
    assert r.denominator == poly_expand([[4, 0, 2], [-5, 0, 7]])
    assert r.gcd == [-5.0, -5.0, 7.0, 7.0]


def _mulp(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            out[i + j] = (out[i + j] + u * v) % p
    return out


def test_factor_mod_p_reconstructs_and_splits_frobenius():
    p = 5
    f = [int(v) % p for v in poly_expand([[1, 1], [2, 0, 1], [3, 1, 1], [1, 2, 0, 1]], [3, 2, 1, 2])]
    r = poly_factor_mod_p(f, p, seed=3)
    prod = [r.unit]
    for g, m in zip(r.factors, r.multiplicities):
        for _ in range(m):
            prod = _mulp(prod, g, p)
    assert prod == f
    # x^p - x splits into all linear factors x - a
    lin = poly_factor_mod_p([0, -1] + [0] * 5 + [1], 7)
    assert lin.factors == [[(-a) % 7, 1] for a in (0, 6, 5, 4, 3, 2, 1)] or sorted(lin.factors) == sorted(
        [[(-a) % 7, 1] for a in range(7)]
    )
    assert poly_factor_mod_p([3, 0, 0, 0, 0, 0, 0, 1], 7).multiplicities == [7]


def test_heat_single_mode_and_quadratic_and_norm():
    r = heat_equation_series(lambda z: math.sin(2 * math.pi * z / 3), 3.0, 0.4, [0.7], [0.5], n_terms=4, n_quad=2000)
    want = math.sin(2 * math.pi * 0.7 / 3) * math.exp(-0.4 * (2 * math.pi / 3) ** 2 * 0.5)
    assert abs(r.u[0][0] - want) <= 1e-12
    q = quadratic_roots(1e-3, 1e4, 3.0)
    for x in q.real:
        assert abs(1e-3 * x * x + 1e4 * x + 3.0) <= 1e-9 * max(1.0, abs(1e4 * x))
    c = quadratic_roots(2.0, 1.0, 5.0)
    assert c.real == [-0.25, -0.25] and abs(c.imag[0] - math.sqrt(39) / 4) <= 1e-15
    n = functional_norm([0, 0.5, 1.0], [0.0, 1.0, 2.0])
    assert abs(n.norm - math.sqrt(0.5 * (0 + 1) / 2 + 0.5 * (1 + 4) / 2)) <= 1e-15
