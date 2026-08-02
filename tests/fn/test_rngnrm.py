"""Tests for rngnrm (range normalization)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rngnrm import rngnrm


def test_rngnrm_maps_to_the_unit_interval_with_endpoints():
    out = rngnrm([2.0, 4.0, 6.0])
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-12)


def test_rngnrm_is_shift_and_scale_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    np.testing.assert_allclose(rngnrm(x), rngnrm(5.0 + 3.0 * x), atol=1e-12)


def test_rngnrm_rejects_constant_input():
    with pytest.raises(ValueError, match="identical"):
        rngnrm([3.0, 3.0, 3.0])
