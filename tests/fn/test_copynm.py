"""Tests for copynm (circular binary segmentation).

Replaces the generated stub, which imported ``copy_number_variant``.
"""

from morie.fn.copynm import cbs_statistic, copynm


def test_the_documented_example_finds_the_buried_aberration():
    x = [0.0] * 40 + [1.5] * 8 + [0.0] * 40
    res = copynm(x, permutations=200, seed=1)
    assert res["changepoints"] == [40, 48]
    assert res["n_segments"] == 3


def test_a_flat_series_has_no_changepoint():
    res = copynm([1.0] * 60, permutations=100, seed=1)
    assert res["changepoints"] == []
    assert res["n_segments"] == 1


def test_a_single_step_is_found():
    x = [0.0] * 30 + [3.0] * 30
    res = copynm(x, permutations=200, seed=1)
    assert res["changepoints"] == [30]


def test_segment_means_are_the_fitted_values():
    x = [0.0] * 30 + [3.0] * 30
    res = copynm(x, permutations=200, seed=1)
    assert abs(res["fitted"][0] - 0.0) < 1e-9
    assert abs(res["fitted"][-1] - 3.0) < 1e-9
    assert len(res["fitted"]) == len(x)


def test_cbs_statistic_maximises_over_arcs():
    x = [0.0] * 20 + [5.0] * 5 + [0.0] * 20
    z, i, j = cbs_statistic(x)
    assert z > 0
    assert 0 <= i < j <= len(x)
    # the maximising arc is the planted aberration
    assert (i, j) == (20, 25)


def test_a_series_too_short_to_test_returns_one_segment():
    # fewer than three points cannot be split; the module returns the
    # whole series rather than raising
    res = copynm([1.0, 2.0], permutations=50, seed=1)
    assert res["changepoints"] == []
    assert res["n_segments"] == 1


def test_validation():
    for call in (lambda: copynm([]),
                 lambda: copynm([1.0] * 10, alpha=0.0),
                 lambda: copynm([1.0] * 10, permutations=0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
