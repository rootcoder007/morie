"""bnfwd: batch normalisation forward pass (Ioffe & Szegedy 2015).

    x_hat = (x - mu) / sqrt(var + eps);   y = gamma * x_hat + beta
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bnfwd import batch_norm_forward as bn


def test_bnfwd_normalised_output_has_zero_mean_and_unit_variance():
    """The defining property. eps makes it approximate, not exact."""
    rng = np.random.default_rng(1601)
    x = rng.normal(5.0, 3.0, (200, 4))
    xh = np.asarray(bn(x)["x_hat"])
    assert np.allclose(xh.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(xh.std(axis=0), 1.0, atol=1e-4)


def test_bnfwd_matches_the_closed_form():
    rng = np.random.default_rng(1607)
    x = rng.standard_normal((50, 3))
    gamma = np.array([2.0, 0.5, -1.0])
    beta = np.array([1.0, 0.0, 3.0])
    eps = 1e-5
    r = bn(x, gamma=gamma, beta=beta, eps=eps)
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    expected = gamma * ((x - mu) / np.sqrt(var + eps)) + beta
    assert np.asarray(r["y"]) == pytest.approx(expected)
    assert np.asarray(r["mu"]) == pytest.approx(mu)
    assert np.asarray(r["var"]) == pytest.approx(var)


def test_bnfwd_default_gamma_one_beta_zero_leaves_x_hat_alone():
    rng = np.random.default_rng(1609)
    r = bn(rng.standard_normal((30, 2)))
    assert np.asarray(r["y"]) == pytest.approx(np.asarray(r["x_hat"]))


def test_bnfwd_is_invariant_to_shifting_the_input():
    """Subtracting the batch mean removes any constant offset exactly."""
    rng = np.random.default_rng(1613)
    x = rng.standard_normal((40, 3))
    assert np.asarray(bn(x + 100.0)["x_hat"]) == pytest.approx(
        np.asarray(bn(x)["x_hat"]), abs=1e-7
    )


def test_bnfwd_scaling_the_input_leaves_x_hat_unchanged_up_to_eps():
    rng = np.random.default_rng(1619)
    x = rng.standard_normal((40, 3))
    assert np.asarray(bn(x * 7.0)["x_hat"]) == pytest.approx(
        np.asarray(bn(x)["x_hat"]), abs=1e-4
    )


def test_bnfwd_constant_feature_does_not_divide_by_zero():
    """Zero variance is exactly what eps is for."""
    x = np.column_stack([np.full(20, 3.0), np.arange(20.0)])
    xh = np.asarray(bn(x)["x_hat"])
    assert np.all(np.isfinite(xh))
    assert np.allclose(xh[:, 0], 0.0)
