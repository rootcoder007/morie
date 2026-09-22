"""Tests for fzkmis.fauzi_kdfe_mise."""

from morie.fn import _array_core as np

from morie.fn.fzkmis import fauzi_kdfe_mise


def test_fzkmis_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    n = int(x.size)
    bandwidth = 0.3
    rfp = 1.0 / (2.0 * np.sqrt(3.141592653589793))
    varint = 1.0 / 6.0
    result = fauzi_kdfe_mise(n, bandwidth, rfp, varint)
    assert isinstance(result, dict)
    assert set(result.keys()) >= {
        "mise",
        "biasterm",
        "varterm",
        "smoothgain",
        "hopt",
        "r1",
        "method",
    }

    mu2 = 1.0
    expected_bias = (bandwidth ** 4) / 4.0 * mu2 ** 2 * rfp
    expected_var = varint / n
    expected_gain = 2.0 * bandwidth / n * result["r1"]
    expected_mise = expected_bias + expected_var - expected_gain
    assert result["biasterm"] == expected_bias
    assert result["varterm"] == expected_var
    assert result["smoothgain"] == expected_gain
    assert result["mise"] == expected_mise


def test_fzkmis_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    n = int(x.size)
    bandwidth = 0.3
    rfp = 1.0 / (2.0 * np.sqrt(3.141592653589793))
    varint = 1.0 / 6.0
    result = fauzi_kdfe_mise(n, bandwidth, rfp, varint)
    assert isinstance(result, dict)
    assert set(result.keys()) >= {"mise", "hopt", "r1", "method"}
    assert isinstance(result["method"], str)

    zero_rfp_result = fauzi_kdfe_mise(n, bandwidth, 0.0, varint)
    assert isinstance(zero_rfp_result["hopt"], float)
    assert zero_rfp_result["hopt"] != zero_rfp_result["hopt"]
    assert zero_rfp_result["biasterm"] == 0.0
