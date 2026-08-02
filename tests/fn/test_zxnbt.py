"""Tests for morie.fn.zxnbt -- Network betweenness spatial"""

from morie.fn import _array_core as np

from morie.fn.zxnbt import network_between


class TestNetworkBetween:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = network_between(data)
        assert result.value is not None

    def test_output_type(self):
        result = network_between(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
