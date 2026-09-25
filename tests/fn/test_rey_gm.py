"""Tests for morie.fn.rey_gm — Gamma GLM regression."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
import pytest

from morie.fn.rey_gm import rey_gm


class TestReyGm:
    """Tests for rey_gm()."""

    def test_basic_gamma_fit(self):
        """Fits on synthetic Gamma-distributed data."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.5 * x)
        y = rng.gamma(shape=5, scale=mu / 5, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_gm(df, y="y", x="x")
        assert result.method == "Gamma GLM"
        assert result.n == n
        assert "intercept" in result.coefficients

    def test_coefficient_direction(self):
        """Positive covariate effect recovered."""
        rng = np.random.default_rng(7)
        n = 300
        x = rng.standard_normal(n)
        mu = np.exp(2.0 + 0.8 * x)
        y = rng.gamma(shape=10, scale=mu / 10, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_gm(df, y="y", x="x")
        assert result.coefficients["x"] > 0

    def test_raises_nonpositive_response(self):
        """Non-positive response raises ValueError."""
        df = pd.DataFrame({"y": [0, 1, 2], "x": [1, 2, 3]})
        with pytest.raises(ValueError, match="positive"):
            rey_gm(df, y="y", x="x")


def _rey_data(zeros=False):
    import math
    n = 80
    x = [math.sin(1.3 * i) for i in range(n)]
    y = [math.exp(1.0 + 0.5 * a) * (0.5 + ((i * 37) % 23) / 22) for i, a in enumerate(x)]
    if zeros:
        y = [0.0 if (i * 7) % 5 == 0 else v for i, v in enumerate(y)]
    return pd.DataFrame({"y": y, "x": x})


def test_matches_statsmodels_gamma_log():
    """Coefficients equal statsmodels 0.15 GLM(Gamma(Log())) on the same data."""
    r = rey_gm(_rey_data(), y="y", x="x")
    assert r.coefficients["intercept"] == pytest.approx(0.989255752842124, rel=1e-9)
    assert r.coefficients["x"] == pytest.approx(0.520087716950179, rel=1e-9)
