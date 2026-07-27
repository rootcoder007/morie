"""Tests for heinz.he_initialization."""

import numpy as np
import pytest

from morie.fn.heinz import he_initialization


def test_heinz_std_is_sqrt_two_over_fan_in():
    """He et al. (2015): std = sqrt(2 / fan_in). That constant is the whole
    point of the initialiser, so check the realised spread against it."""
    fan_in, fan_out = 512, 256
    r = he_initialization(fan_in, fan_out, seed=0)
    W = np.asarray(r["W"], dtype=float)
    # Weights are stored (fan_out, fan_in), the row-of-outputs convention that
    # PyTorch's nn.Linear also uses.
    assert W.shape == (fan_out, fan_in)

    want = np.sqrt(2.0 / fan_in)
    # 131072 draws, so the sample std is tight around the target.
    assert float(W.std()) == pytest.approx(want, rel=0.02)
    assert float(W.mean()) == pytest.approx(0.0, abs=5 * want / np.sqrt(W.size))


def test_heinz_scales_with_fan_in_not_fan_out():
    """Doubling fan_in must shrink the spread by sqrt(2); changing fan_out
    must not move it. This separates He from Xavier/Glorot."""
    a = np.asarray(he_initialization(256, 64, seed=1)["W"], dtype=float).std()
    b = np.asarray(he_initialization(512, 64, seed=1)["W"], dtype=float).std()
    c = np.asarray(he_initialization(256, 128, seed=1)["W"], dtype=float).std()
    assert b == pytest.approx(a / np.sqrt(2), rel=0.05)
    assert c == pytest.approx(a, rel=0.05)


def test_heinz_uniform_mode_has_the_matching_variance():
    r = he_initialization(400, 100, seed=3, mode="uniform")
    W = np.asarray(r["W"], dtype=float)
    assert float(W.std()) == pytest.approx(np.sqrt(2.0 / 400), rel=0.05)


def test_heinz_is_reproducible_and_validates_input():
    x = np.asarray(he_initialization(64, 32, seed=7)["W"], dtype=float)
    y = np.asarray(he_initialization(64, 32, seed=7)["W"], dtype=float)
    np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match="fan_in"):
        he_initialization(0, 10)
    with pytest.raises(ValueError, match="mode"):
        he_initialization(8, 8, mode="lognormal")
