"""Tests for morie.fn.nmocm -- OC Coombs mesh"""

from morie.fn import _array_core as np

from morie.fn.nmocm import oc_coombs_mesh


class TestOcCoombsMesh:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = oc_coombs_mesh(data)
        assert result.value is not None

    def test_output_type(self):
        result = oc_coombs_mesh(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
