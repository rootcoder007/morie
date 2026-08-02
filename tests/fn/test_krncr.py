"""Tests for morie.fn.krncr -- Kronecker product."""

from morie.fn import _array_core as np

from morie.fn._containers import DescriptiveResult
from morie.fn.krncr import krncr, kronecker


class TestKrncr:
    def test_alias(self):
        assert krncr is kronecker

    def test_identity(self):
        r = kronecker(np.eye(2), np.eye(2))
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.extra["matrix"], np.eye(4))
        assert r.value == 16

    def test_shape(self):
        A = np.ones((2, 3))
        B = np.ones((4, 5))
        r = kronecker(A, B)
        assert r.extra["shape"] == (8, 15)
