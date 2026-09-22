"""Tests for aitbal.aitchison_balance."""

from morie.fn import _array_core as np

from morie.fn.aitbal import aitchison_balance


def test_aitbal_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 10.0, 6)
    num_idx = [0, 1]
    den_idx = [2, 3, 4, 5]

    result = aitchison_balance(x, num_idx, den_idx)

    assert "balance" in result
    assert "normalizer" in result
    assert "geometric_mean_num" in result
    assert "geometric_mean_den" in result

    x_arr = np.asarray(x, dtype=float)
    num = np.asarray(num_idx, dtype=int)
    den = np.asarray(den_idx, dtype=int)
    r = num.size
    s = den.size
    gn = np.exp(np.log(x_arr[num]).mean())
    gd = np.exp(np.log(x_arr[den]).mean())
    expected_norm = np.sqrt(r * s / (r + s))
    expected_bal = expected_norm * np.log(gn / gd)

    assert abs(float(result["balance"]) - float(expected_bal)) < 1e-12
    assert abs(float(result["normalizer"]) - float(expected_norm)) < 1e-12
    assert abs(float(result["geometric_mean_num"]) - float(gn)) < 1e-12
    assert abs(float(result["geometric_mean_den"]) - float(gd)) < 1e-12

    scaled = aitchison_balance(x * 7.3, num_idx, den_idx)["balance"]
    assert abs(float(scaled) - float(result["balance"])) < 1e-12


def test_aitbal_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 10.0, 4)
    num_idx = [0]
    den_idx = [1, 2, 3]

    result = aitchison_balance(x, num_idx, den_idx)
    assert isinstance(result.balance, float)
    assert result.balance == result["balance"]

    up = aitchison_balance([40.0, 25.0, 35.0], [0], [1, 2])["balance"]
    base = aitchison_balance([20.0, 30.0, 50.0], [0], [1, 2])["balance"]
    assert float(up) > float(base)
