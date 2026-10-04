"""Test gf2_matrix_add."""

from morie.fn import _array_core as np
from morie.fn._containers import DescriptiveResult
from morie.fn.gf2ad import gf2_matrix_add


class TestGf2MatrixAdd:
    def test_basic(self):
        I_ = np.eye(3, dtype=int)
        result = gf2_matrix_add(a=I_, b=I_)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        I_ = np.eye(3, dtype=int)
        result = gf2_matrix_add(a=I_, b=I_)
        assert "result" in result.extra

    def test_self_add_is_zero(self):
        I_ = np.eye(3, dtype=int)
        result = gf2_matrix_add(a=I_, b=I_)
        r = np.asarray(result.extra["result"])
        np.testing.assert_array_equal(r, np.zeros((3, 3), dtype=int))
