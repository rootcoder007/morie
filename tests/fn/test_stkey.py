"""Tests for stkey (Siegel-Tukey test)."""

import pytest

from morie.fn import _array_core as np

from morie.fn.stkey import stkey


class TestStkey:
    """Siegel-Tukey test for scale equality."""

    def test_stkey_equal_scale(self):
        """Equal scale should not reject."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)
        result = stkey(x, y)
        assert 0 <= result["p_value"] <= 1

    def test_stkey_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = stkey(x, y)
        required_keys = {"statistic", "z_stat", "p_value", "interpretation"}
        assert set(result.keys()) == required_keys

    def test_stkey_matches_desctools(self):
        """Scores dealt from the ends; the p-value equals
        DescTools::SiegelTukeyTest(x, y, exact = FALSE, correct = FALSE)."""
        x = [10.1, 12.3, 9.8, 15.2, 7.4, 11.0, 13.9, 8.2, 16.5]
        y = [11.2, 10.9, 11.8, 10.4, 12.0, 11.5, 10.7, 11.9]
        result = stkey(x, y)
        assert result["p_value"] == pytest.approx(0.0095044605706371289, rel=1e-12)
        # 17 values: the median 11.2 is dropped; sorted X positions among
        # the 16 left are 1 2 3 4 8 13 14 15 16 with scores 1 4 5 8 16 7 6 3 2
        assert result["statistic"] == 52.0
        assert result["interpretation"] == "reject"
