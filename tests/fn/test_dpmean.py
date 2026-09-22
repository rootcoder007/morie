"""Tests for dpmean.dp_mean."""

from morie.fn import _array_core as np

from morie.fn.dpmean import dp_mean


def test_dpmean_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    a = 0.0
    b = 1.0
    epsilon = 1.0
    result = dp_mean(x, a, b, epsilon, seed=0)
    assert isinstance(result, dict)
    # Documented return-key names per the function's docstring/payload.
    assert "release" in result
    assert "true_mean" in result
    assert "sensitivity" in result
    assert "noise_scale" in result
    assert "clipped_fraction" in result
    assert "epsilon_sum" in result
    assert "epsilon_count" in result
    assert "estimate" not in result
    assert "statistic" not in result

    # With n public and a seed, the output is deterministic and known.
    n = 100
    sens_expected = (b - a) / n
    scale_expected = sens_expected / epsilon
    assert abs(float(result["sensitivity"]) - sens_expected) < 1e-12
    assert abs(float(result["noise_scale"]) - scale_expected) < 1e-12
    assert abs(float(result["epsilon_sum"]) - epsilon) < 1e-12
    assert float(result["epsilon_count"]) == 0.0
    assert 0.0 <= float(result["clipped_fraction"]) <= 1.0


def test_dpmean_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    a = 0.0
    b = 1.0
    epsilon = 1.0
    result = dp_mean(x, a, b, epsilon, seed=0)
    assert isinstance(result, dict)
    assert "release" in result
    assert "true_mean" in result

    # known_n=False splits the budget: epsilon_sum + epsilon_count == epsilon.
    r_priv = dp_mean(x, a, b, epsilon, seed=0, known_n=False, split=0.5)
    assert abs(float(r_priv["epsilon_sum"]) - 0.5) < 1e-12
    assert abs(float(r_priv["epsilon_count"]) - 0.5) < 1e-12
    assert abs(
        float(r_priv["epsilon_sum"]) + float(r_priv["epsilon_count"]) - epsilon
    ) < 1e-12
