"""Tests for johsn.johansen_cointegration."""

import numpy as np
import pytest

from morie.fn.johsn import johansen_cointegration


def _cointegrated(seed, n=400):
    """x2 = x1 + stationary error, x1 a random walk: rank 1 by construction."""
    rng = np.random.default_rng(seed)
    x1 = np.cumsum(rng.standard_normal(n))
    x2 = x1 + rng.standard_normal(n) * 0.5
    return np.column_stack([x1, x2])


def _independent_walks(seed, n=400):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, 2)), axis=0)


def test_johsn_finds_rank_one_in_a_cointegrated_pair():
    """Measured: rank 1 detected on seeds 1..3 at the 95 percent level."""
    for s in (1, 2, 3):
        r = johansen_cointegration(_cointegrated(s))
        assert int(r["rank"]) >= 1, f"seed {s}"


def test_johsn_finds_no_cointegration_between_independent_walks():
    """Measured 0/3 spurious detections on seeds 11..13."""
    ranks = [int(johansen_cointegration(_independent_walks(s))["rank"]) for s in (11, 12, 13)]
    assert sum(1 for k in ranks if k > 0) <= 1


def test_johsn_trace_statistics_decrease_and_match_critvals_shape():
    r = johansen_cointegration(_cointegrated(4))
    trace = np.asarray(r["trace_stat"], dtype=float)
    cvt = np.asarray(r["crit_values"], dtype=float)
    assert trace.shape[0] == 2 and cvt.shape[0] == 2
    assert trace[0] > trace[1]  # H0: r=0 carries the largest statistic


def test_johsn_rejects_short_series():
    with pytest.raises(ValueError, match="T>=20"):
        johansen_cointegration(np.random.default_rng(0).standard_normal((10, 2)))
