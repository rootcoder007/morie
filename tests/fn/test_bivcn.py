"""Tests for bivcn -- the front-end over anmod."""

from morie.fn import _array_core as np
import pytest

from morie.fn.anmod import additive_noise_model
from morie.fn.bivcn import bivariate_causal_test


def _cubic(seed=1, n=250):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    return x, x**3 + rng.normal(0, 0.5, n)


def test_front_end_matches_the_canonical_implementation():
    x, y = _cubic()
    a = additive_noise_model(x, y, B=60, seed=5)
    b = bivariate_causal_test(x, y, B=60, seed=5)
    assert a["direction"] == b["direction"]
    assert a["hsic_xy"] == b["hsic_xy"]


def test_an_unimplemented_regressor_is_refused_not_ignored():
    """Silently using a different smoother would misreport what was run."""
    x, y = _cubic()
    with pytest.raises(ValueError, match="not implemented"):
        bivariate_causal_test(x, y, regressor="gp")


def test_it_does_not_carry_a_second_implementation():
    import inspect

    from morie.fn import bivcn

    assert "anmod" in inspect.getsource(bivcn)
