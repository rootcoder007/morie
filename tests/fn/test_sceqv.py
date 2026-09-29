"""Tests for morie.fn.sceqv — score equating."""

import pytest

from morie.fn.sceqv import score_equate


class TestScoreEquate:
    def test_linear(self, rng):
        s1 = rng.standard_normal(200) * 10 + 50
        s2 = rng.standard_normal(200) * 8 + 45
        result = score_equate(s1, s2, method="linear")
        assert result["method"] == "linear"
        assert len(result["concordance"]) > 0

    def test_equipercentile(self, rng):
        s1 = rng.standard_normal(200) * 10 + 50
        s2 = rng.standard_normal(200) * 8 + 45
        result = score_equate(s1, s2, method="equipercentile")
        assert result["method"] == "equipercentile"
        assert len(result["concordance"]) > 0

    def test_stats_present(self, rng):
        s1 = rng.standard_normal(100)
        s2 = rng.standard_normal(100)
        result = score_equate(s1, s2)
        assert "form1_stats" in result
        assert "form2_stats" in result
        assert result["form1_stats"]["n"] == 100

    def test_invalid_method(self, rng):
        with pytest.raises(ValueError):
            score_equate(rng.standard_normal(50), rng.standard_normal(50), method="bad")

    def test_identical_forms(self, rng):
        s = rng.standard_normal(200)
        result = score_equate(s, s, method="linear")
        # Same form -> concordance should map each score to approximately itself
        for score_in, score_out in result["concordance"].items():
            assert abs(score_in - score_out) < 0.01


def test_linear_and_equipercentile_concordance():
    import math

    s1 = [50.0, 55.0, 60.0, 65.0, 70.0, 45.0]
    s2 = [40.0, 42.0, 50.0, 48.0, 46.0, 44.0]

    def ms(v):
        m = sum(v) / len(v)
        return m, math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1))

    m1, sd1 = ms(s1)
    m2, sd2 = ms(s2)
    lin = score_equate(s1, s2, method="linear")["concordance"]
    for x in s2:
        assert lin[x] == pytest.approx(sd1 / sd2 * (x - m2) + m1, rel=1e-13)
    eq = score_equate(s1, s2)["concordance"]
    xs = sorted(s1)
    pr = sum(v <= 46.0 for v in s2) / 6 * 100
    h = 5 * pr / 100
    lo = int(h)
    assert eq[46.0] == pytest.approx(xs[lo] + (h - lo) * (xs[min(lo + 1, 5)] - xs[lo]), rel=1e-13)
