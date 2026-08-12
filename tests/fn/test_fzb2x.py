"""Tests for fzb2x (b_2 bias coefficient of the kernel distribution
function estimator).

Replaces the generated stub, which imported ``fauzi_b2_coefficient``.
"""

from morie.fn.fzb2x import kdfb2


def test_the_coefficient_is_proportional_to_the_density_derivative():
    # b_2 = mu2 * f'(x) / 2 in the usual expansion, so it is linear in
    # both arguments
    a = kdfb2(0.4, mu2=1.0)["estimate"]
    b = kdfb2(0.8, mu2=1.0)["estimate"]
    assert abs(b - 2.0 * a) < 1e-12


def test_it_is_linear_in_mu2():
    a = kdfb2(0.4, mu2=1.0)["estimate"]
    b = kdfb2(0.4, mu2=3.0)["estimate"]
    assert abs(b - 3.0 * a) < 1e-12


def test_a_flat_density_gives_no_bias_term():
    assert abs(kdfb2(0.0)["estimate"]) < 1e-15


def test_the_sign_follows_the_derivative():
    assert kdfb2(1.0)["estimate"] > 0
    assert kdfb2(-1.0)["estimate"] < 0
    assert kdfb2(1.0)["mu2"] == 1.0
