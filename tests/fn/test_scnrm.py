"""Tests for morie.fn.scnrm — score norms."""

from morie.fn.scnrm import score_norms


class TestScoreNorms:
    def test_returns_dict(self, rng):
        scores = rng.standard_normal(200)
        result = score_norms(scores)
        assert isinstance(result, dict)
        assert "n" in result
        assert "mean" in result
        assert "percentiles" in result

    def test_default_percentiles(self, rng):
        scores = rng.standard_normal(200)
        result = score_norms(scores)
        assert set(result["percentiles"].keys()) == {5, 25, 50, 75, 95}

    def test_custom_percentiles(self, rng):
        scores = rng.standard_normal(200)
        result = score_norms(scores, percentiles=[10, 50, 90])
        assert set(result["percentiles"].keys()) == {10, 50, 90}

    def test_n_correct(self, rng):
        scores = rng.standard_normal(150)
        result = score_norms(scores)
        assert result["n"] == 150

    def test_percentile_ordering(self, rng):
        scores = rng.standard_normal(500)
        result = score_norms(scores)
        p = result["percentiles"]
        assert p[5] <= p[25] <= p[50] <= p[75] <= p[95]


def test_norm_table_recomputed():
    import math

    import pytest

    s = [12.0, 15.0, 9.0, 20.0, 17.0, 11.0, float("nan"), 14.0]
    v = [t for t in s if t == t]
    n = len(v)
    m = sum(v) / n
    xs = sorted(v)
    h = (n - 1) * 0.25
    q25 = xs[int(h)] + (h - int(h)) * (xs[int(h) + 1] - xs[int(h)])
    r = score_norms(s, percentiles=[25])
    assert r["n"] == 7
    assert r["mean"] == pytest.approx(m, rel=1e-14)
    assert r["sd"] == pytest.approx(math.sqrt(sum((t - m) ** 2 for t in v) / (n - 1)), rel=1e-13)
    assert r["percentiles"][25] == pytest.approx(q25, rel=1e-13)
