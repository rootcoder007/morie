"""Tests for morie.fn.apres -- aggregate PRE."""

from morie.fn.apres import apre_statistic, apres


def test_alias():
    assert apres is apre_statistic


def test_smoke():
    pre_vals = [0.8, 0.6, 0.9, 0.7]
    r = apre_statistic(pre_vals)
    assert r.name == "apre_statistic"
    assert abs(r.value - 0.75) < 1e-10


def test_extra_fields():
    r = apre_statistic([0.5, 0.5])
    assert "median_pre" in r.extra
    assert "n_roll_calls" in r.extra
    assert r.extra["n_roll_calls"] == 2


def test_apre_is_the_minority_weighted_pooled_pre():
    """APRE = sum(m_j - e_j)/sum(m_j), recomputed from minority and error counts."""
    import pytest

    minority = [40, 10, 25, 3]
    errors = [8, 5, 10, 3]
    pre = [(m - e) / m for m, e in zip(minority, errors)]
    r = apre_statistic(pre, minority=minority)
    assert r.value == pytest.approx(sum(m - e for m, e in zip(minority, errors)) / sum(minority), rel=1e-14)
    assert r.extra["weighted"] is True
    assert r.extra["mean_pre"] == pytest.approx(sum(pre) / 4, rel=1e-14)
    assert r.extra["median_pre"] == pytest.approx(sorted(pre)[1] / 2 + sorted(pre)[2] / 2, rel=1e-14)
