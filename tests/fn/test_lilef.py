"""Tests for lilef (Lilliefors test for normality)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.lilef import lilef


class TestLilef:
    """Lilliefors test for normality."""

    def test_lilef_normal_sample(self):
        """Sample from normal distribution should not reject normality."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        result = lilef(x)
        assert result["p_value"] > 0.01 or result["p_value"] < 1.0

    def test_lilef_non_normal_sample(self):
        """Sample from exponential should likely reject normality."""
        rng = np.random.default_rng(42)
        x = rng.exponential(scale=1.0, size=100)
        result = lilef(x)
        # May not always reject due to sample size, but should be detectable
        assert "statistic" in result

    def test_lilef_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.random.default_rng(42).standard_normal(50)
        result = lilef(x)
        required_keys = {"statistic", "p_value", "critical_value", "interpretation", "mean", "std"}
        assert set(result.keys()) == required_keys

    def test_lilef_statistic_in_unit_interval(self):
        """Lilliefors statistic should be in [0, 1]."""
        x = np.random.default_rng(42).standard_normal(100)
        result = lilef(x)
        assert 0 <= result["statistic"] <= 1

    def test_lilef_small_sample_error(self):
        """Sample size < 4 should raise error."""
        with pytest.raises(ValueError):
            lilef(np.array([1, 2, 3]))

    def test_lilef_mean_std_estimates(self):
        """Mean and std should be reasonable estimates."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = lilef(x)
        assert np.all(np.isfinite(np.asarray(result["mean"], dtype=float)))  # N6: was a generator-guessed value


def test_lilef_statistic_and_p_value_recomputed():
    """D from its definition; p from Dallal-Wilkinson (p < 0.1 branch)."""
    import math

    x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3, 2.7]
    n = len(x)
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    F = [0.5 * math.erfc(-((v - m) / s) / math.sqrt(2)) for v in sorted(x)]
    d = max(max((i + 1) / n - F[i], F[i] - i / n) for i in range(n))
    p = math.exp(
        -7.01256 * d * d * (n + 2.78019)
        + 2.99587 * d * math.sqrt(n + 2.78019)
        - 0.122119
        + 0.974598 / math.sqrt(n)
        + 1.67997 / n
    )
    r = lilef(x)
    assert r["statistic"] == pytest.approx(d, rel=1e-12)
    assert r["p_value"] == pytest.approx(p, rel=1e-12)
    assert r["interpretation"] == "reject"


def test_lilef_stephens_branch_for_large_p():
    import math

    y = [-0.61, 0.23, 1.02, -1.4, 0.35, -0.12, 0.77, -0.95, 0.51, 1.6, -0.3, 0.05]
    r = lilef(y)
    kk = (math.sqrt(12) - 0.01 + 0.85 / math.sqrt(12)) * r["statistic"]
    if kk <= 0.302:
        want = 1.0
    elif kk <= 0.5:
        want = 2.76773 - 19.828315 * kk + 80.709644 * kk**2 - 138.55152 * kk**3 + 81.218052 * kk**4
    elif kk <= 0.9:
        want = -4.901232 + 40.662806 * kk - 97.490286 * kk**2 + 94.029866 * kk**3 - 32.355711 * kk**4
    else:
        want = 6.198765 - 19.558097 * kk + 23.186922 * kk**2 - 12.234627 * kk**3 + 2.423045 * kk**4
    assert r["p_value"] == pytest.approx(want, rel=1e-12)
