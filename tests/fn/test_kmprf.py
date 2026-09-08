"""Tests for kmprf.kamath_prefix_lm_mask."""

from morie.fn import _array_core as np
import pytest

from morie.fn.kmprf import kamath_prefix_lm_mask


def test_kmprf_basic():
    m = kamath_prefix_lm_mask(2, 4)["mask"]
    assert np.array_equal(
        m,
        np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
            dtype=bool,
        ),
    )


def test_kmprf_edge():
    assert np.array_equal(kamath_prefix_lm_mask(0, 3)["mask"], np.tril(np.ones((3, 3), bool)))
    assert kamath_prefix_lm_mask(3, 3)["mask"].all()
    with pytest.raises(ValueError):
        kamath_prefix_lm_mask(4, 3)
