"""Tests for Thomas process."""

from morie.fn.sgthm import sgthm


def test_sgthm_smoke():
    r = sgthm(5.0, 10.0, 0.5, (0, 10, 0, 10), seed=42)
    assert r.name == "thomas_process"
    assert "points" in r.extra
    assert r.extra["n_points"] > 0
    # the headline value is the retained point count
    assert r.value == float(r.extra["n_points"])
    assert len(r.extra["points"]) == r.extra["n_points"]
    assert r.extra["expected_intensity"] == 5.0 * 10.0
    # every retained offspring lies inside the window
    for x, y in r.extra["points"]:
        assert 0.0 <= float(x) <= 10.0
        assert 0.0 <= float(y) <= 10.0


def test_sgthm_is_reproducible():
    a = sgthm(5.0, 10.0, 0.5, (0, 10, 0, 10), seed=7)
    b = sgthm(5.0, 10.0, 0.5, (0, 10, 0, 10), seed=7)
    assert a.extra["n_points"] == b.extra["n_points"]
    assert a.extra["n_parents"] == b.extra["n_parents"]
    for (ax, ay), (bx, by) in zip(a.extra["points"], b.extra["points"]):
        assert float(ax) == float(bx)
        assert float(ay) == float(by)


def test_sgthm_no_offspring():
    """mu = 0 leaves parents but no offspring, so the pattern is empty."""
    r = sgthm(5.0, 0.0, 0.5, (0, 10, 0, 10), seed=3)
    assert r.extra["n_points"] == 0
    assert r.value == 0.0
    assert r.extra["n_parents"] > 0


def test_cheatsheet():
    from morie.fn.sgthm import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
    assert cs.startswith("thomas_process")
