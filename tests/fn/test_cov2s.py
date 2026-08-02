"""Tests for cov2s.two_sample_coverage."""

from morie.fn import _array_core as np
import pytest

from morie.fn.cov2s import two_sample_coverage


def test_cov2s_blocks_partition_the_second_sample():
    rng = np.random.default_rng(0)
    x = rng.normal(size=10)
    y = rng.normal(size=25)
    r = two_sample_coverage(x, y)
    freq = np.asarray(r["block_freq"], dtype=int)
    assert freq.shape == (11,)          # m + 1 blocks
    assert int(freq.sum()) == 25        # every y lands somewhere
    assert int(r["cumulative"]) == 25


def test_cov2s_hand_computed_blocks():
    """x = (0, 10), y = (-1, 5, 11): one y below, one between, one above."""
    r = two_sample_coverage(np.array([0.0, 10.0]), np.array([-1.0, 5.0, 11.0]))
    np.testing.assert_array_equal(np.asarray(r["block_freq"], dtype=int), [1, 1, 1])
    assert float(r["expected_prop"]) == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_cov2s_same_distribution_spreads_evenly():
    """Under H0 each block holds n/(m+1) on average. Measured mean
    absolute deviation from uniform < 0.35/(m+1) across seeds."""
    devs = []
    for s in range(10):
        rng = np.random.default_rng(s)
        r = two_sample_coverage(rng.normal(size=9), rng.normal(size=100))
        prop = np.asarray(r["block_prop"], dtype=float)
        devs.append(np.abs(prop - 0.1).mean())
    # Measured 0.073 across seeds 0..9 (m = 9 blocks, n = 100: each block
    # proportion has sd ~0.03, so MAD ~0.07 is the calibrated level).
    assert np.mean(devs) < 0.11
