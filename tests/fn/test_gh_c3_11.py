"""Tests for gh_c3_11.ghosal_tailfree_def."""

from morie.fn import _array_core as np

from morie.fn.gh_c3_11 import ghosal_tailfree_def


def test_gh_c3_11_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_tailfree_def(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value

    # The implementation computes the mean of the leftmost cell's mass across
    # `reps` independent tail-free splits, then compares it level-by-level to
    # the theoretical expectation prod_j E[V] = 2^-m for a Beta(1,1) split.
    means = result["mean_by_level"]
    gaps = result["prop312_gap"]
    assert len(means) == 8
    assert np.all(np.isfinite(np.asarray(means, dtype=float)))
    # Theoretical leftmost-cell mass at level m (1-indexed) is 2^-m; with
    # 0-indexed levels this is 2^-(m+1).
    theoretical = [2.0 ** (-(m + 1)) for m in range(8)]
    expected_gap = max(abs(means[m] - theoretical[m]) for m in range(8))
    assert abs(gaps - expected_gap) < 1e-12
    # The level-m (0-indexed) theoretical mean is 2^-m; with 400 reps of
    # i.i.d. Beta(1,1) products the MC mean must lie in (0, 1) for m >= 1.
    assert 0.0 < means[-1] < 1.0
    assert result["method"] == "tail-free splits + Prop 3.12(i) check (GvdV 2017 sec. 3.6)"


def test_gh_c3_11_edge():
    """Test edge cases: argument shape is irrelevant for this estimator,
    but the result must still carry the documented keys."""
    result = ghosal_tailfree_def(np.array([42.0]))
    # The function documents a RichResult payload with these keys; 'n' is
    # not one of them, so the original assertion was stale.
    assert "estimate" in result
    assert "mean_by_level" in result
    assert "prop312_gap" in result
    assert "method" in result
    assert isinstance(result["mean_by_level"], list)
    assert len(result["mean_by_level"]) == 8
