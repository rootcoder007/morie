"""Tests for morie.fn.svord2."""

from morie.fn import _array_core as np

from morie.fn.svord2 import svord2


class TestSvord2:
    def test_basic(self):
        result = svord2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert result is not None
        assert result.statistic is not None
        assert isinstance(result.statistic, float)

    def test_returns_spatial_result(self):
        result = svord2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert hasattr(result, "statistic")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svord2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert np.isfinite(result.statistic)

    def test_name_string(self):
        result = svord2(np.array([0.5, 0.5]), np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
        assert result.name == "Ordered Logit Spatial Model"


def test_svord2_conditional_logit_values():
    """P_j = exp(-beta d_j) / sum_k exp(-beta d_k), recomputed here."""
    import math
    voter = [0.2, -0.1]
    cands = [[1.0, 1.0], [0.0, 0.0], [-2.0, 0.5], [0.5, -0.4]]
    beta = 1.7
    r = svord2(voter, cands, beta=beta)
    d = [math.dist(voter, c) for c in cands]
    e = [math.exp(-beta * v) for v in d]
    want = [v / sum(e) for v in e]
    for got, ref in zip(r.extra["probabilities"], want):
        assert abs(got - ref) < 1e-15
    assert abs(r.statistic - want[d.index(min(d))]) < 1e-15
