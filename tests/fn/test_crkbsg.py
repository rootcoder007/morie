"""Tests for crkbsg.cokriging."""

from morie.fn import _array_core as np

from morie.fn.crkbsg import cokriging


def test_crkbsg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    d = 2
    coords = rng.uniform(0, 1, (n, d))
    y = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    s_predict = rng.uniform(0, 1, (5, d))
    cross_variogram = {
        'model': 'spherical',
        'range': 1.0,
        'b11': 1.0,
        'b22': 1.0,
        'b12': 0.5,
        'nugget11': 0.0,
        'nugget22': 0.0,
        'nugget12': 0.0,
    }
    result = cokriging(coords, y, z, s_predict, cross_variogram)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert len(result["prediction"]) == 5


def test_crkbsg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    d = 2
    coords = rng.uniform(0, 1, (n, d))
    y = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    s_predict = rng.uniform(0, 1, (1, d))
    cross_variogram = {
        'model': 'spherical',
        'range': 1.0,
        'b11': 1.0,
        'b22': 1.0,
        'b12': 0.5,
        'nugget11': 0.0,
        'nugget22': 0.0,
        'nugget12': 0.0,
    }
    result = cokriging(coords, y, z, s_predict, cross_variogram)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert len(result["prediction"]) == 1
