"""Tests for morie.fn.lacjmc."""

from morie.fn.lacjmc import lacjmc


def ring(n):
    return [[1.0 if abs(i - j) in (1, n - 1) else 0.0 for j in range(n)] for i in range(n)]


class TestLacjmc:
    y_binary = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0]

    def test_basic(self):
        result = lacjmc(self.y_binary, ring(20), 9)
        assert result.statistic == 5.0 and len(result.extra["simulated"]) == 9

    def test_returns_spatial_result(self):
        result = lacjmc(self.y_binary, ring(20), 9)
        assert hasattr(result, "statistic") and 0.1 <= result.p_value <= 1

    def test_statistic_numeric(self):
        result = lacjmc(self.y_binary, ring(20), 199, seed=3)
        # permutation mean near the exact E[BB] = 20 * 10 * 9 / (20 * 19) under non-free sampling
        assert abs(result.expected - 20 * 10 * 9 / (20 * 19)) < 0.4
