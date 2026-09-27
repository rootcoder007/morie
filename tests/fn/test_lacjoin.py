"""Tests for morie.fn.lacjoin."""

from morie.fn.lacjoin import lacjoin


def ring(n):
    return [[1.0 if abs(i - j) in (1, n - 1) else 0.0 for j in range(n)] for i in range(n)]


class TestLacjoin:
    y_binary = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]

    def test_basic(self):
        result = lacjoin(self.y_binary, ring(20))
        y = self.y_binary
        assert result.statistic == sum(1 for i in range(20) if y[i] and y[(i + 1) % 20])

    def test_returns_spatial_result(self):
        result = lacjoin(self.y_binary, ring(20))
        assert hasattr(result, "statistic") and 0 < result.p_value < 1

    def test_statistic_numeric(self):
        result = lacjoin(self.y_binary, ring(20))
        e = result.extra
        assert e["BB"] + e["WW"] + e["BW"] == 20
