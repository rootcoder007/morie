"""Tests for gh_c6_15.ghosal_martg_consist."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_15 import ghosal_martg_consist


def test_gh_c6_15_basic():
    """Test basic functionality with valid squared Hellinger distances in [0, 1]."""
    rng = np.random.default_rng(42)
    n = 100
    # squared Hellinger distances must lie in [0, 1]
    dh2 = rng.uniform(0.0, 1.0, n)
    result = ghosal_martg_consist(dh2)

    assert isinstance(result, dict)
    # documented return keys
    for key in ("cesaro", "final", "tail_mean", "lemma652_sum", "summable", "n"):
        assert key in result

    # n is reported
    assert result["n"] == float(n)

    # independent recomputation of the Cesaro averages
    run_sum = 0.0
    cesaro_expected = []
    for i, v in enumerate(dh2):
        run_sum += v
        cesaro_expected.append(run_sum / (i + 1))
    assert result["cesaro"] == cesaro_expected

    # final is the last Cesaro average
    assert result["final"] == cesaro_expected[-1]

    # tail_mean is the average over the last half
    half = n // 2
    tail_mean_expected = sum(dh2[half:]) / (n - half)
    assert result["tail_mean"] == tail_mean_expected

    # no variances given -> lemma652_sum is NaN and summable is NaN
    assert result["lemma652_sum"] != result["lemma652_sum"]  # NaN != NaN
    assert result["summable"] != result["summable"]


def test_gh_c6_15_with_variances():
    """Test with variances: the Lemma 6.52 partial sum is computed."""
    rng = np.random.default_rng(43)
    n = 50
    dh2 = rng.uniform(0.0, 1.0, n)
    variances = rng.uniform(0.0, 1.0, n)

    result = ghosal_martg_consist(dh2, variances=variances)

    assert isinstance(result, dict)
    # independent recomputation of the Lemma 6.52 sum
    expected_ls = sum(variances[i] / ((i + 1) ** 2) for i in range(n))
    assert result["lemma652_sum"] == expected_ls
    # a finite sum is summable
    assert result["summable"] == 1.0


def test_gh_c6_15_edge():
    """Test edge cases with minimal valid input."""
    dh2 = [0.5]
    result = ghosal_martg_consist(dh2)
    assert isinstance(result, dict)
    assert result["n"] == 1.0
    # Cesaro average of a single value equals the value itself
    assert result["final"] == 0.5
    # tail_mean over the last half (which is the whole array here)
    assert result["tail_mean"] == 0.5
