"""Tests for morie.fn.rey_tw — Tweedie regression."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
import pytest

from morie.fn.rey_tw import rey_tw


class TestReyTw:
    """Tests for rey_tw()."""

    def test_basic_tweedie_fit(self):
        """Fits on synthetic non-negative data."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        y = rng.gamma(shape=2, scale=mu / 2, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_tw(df, y="y", x="x", power=1.5)
        assert result.method == "Tweedie GLM"
        assert result.n == n
        assert result.extra["power"] == 1.5

    def test_coefficient_sign(self):
        """Positive coefficient recovered."""
        rng = np.random.default_rng(7)
        n = 300
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.7 * x)
        y = rng.gamma(shape=3, scale=mu / 3, size=n)
        df = pd.DataFrame({"y": y, "x": x})
        result = rey_tw(df, y="y", x="x", power=1.5)
        assert result.coefficients["x"] > 0

    def test_raises_invalid_power(self):
        """Power outside (1,2) raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
        with pytest.raises(ValueError, match="power"):
            rey_tw(df, y="y", x="x", power=0.5)


def _rey_data(zeros=False):
    import math
    n = 80
    x = [math.sin(1.3 * i) for i in range(n)]
    y = [math.exp(1.0 + 0.5 * a) * (0.5 + ((i * 37) % 23) / 22) for i, a in enumerate(x)]
    if zeros:
        y = [0.0 if (i * 7) % 5 == 0 else v for i, v in enumerate(y)]
    return pd.DataFrame({"y": y, "x": x})


def test_matches_statsmodels_tweedie():
    """Coefficients, standard errors, Pearson dispersion and deviance equal
    statsmodels 0.15 GLM(Tweedie(var_power=1.5, link=Log())), zeros included."""
    r = rey_tw(_rey_data(zeros=True), y="y", x="x", power=1.5)
    assert r.coefficients["intercept"] == pytest.approx(0.797160171169335, rel=1e-12)
    assert r.coefficients["x"] == pytest.approx(0.3165400378581693, rel=1e-12)
    assert r.se["intercept"] == pytest.approx(0.07575544185130777, rel=1e-12)
    assert r.se["x"] == pytest.approx(0.10702372875069331, rel=1e-12)
    assert r.extra["phi"] == pytest.approx(0.6773500306791018, rel=1e-12)
    assert r.extra["deviance"] == pytest.approx(124.94337043069578, rel=1e-12)
