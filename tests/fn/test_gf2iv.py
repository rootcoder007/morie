"""Test gf2_matrix_inv."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.gf2iv import gf2_matrix_inv


class TestGf2MatrixInv:
    def test_basic(self):
        I_ = np.eye(3, dtype=int)
        result = gf2_matrix_inv(a=I_)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        I_ = np.eye(3, dtype=int)
        result = gf2_matrix_inv(a=I_)
        assert "inverse" in result.extra

    def test_identity_inverse(self):
        I_ = np.eye(3, dtype=int)
        result = gf2_matrix_inv(a=I_)
        r = np.asarray(result.extra["inverse"])
        np.testing.assert_array_equal(r, I_)
