"""Tests for gb1021.gibbons_k_median_test."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gb1021 import gibbons_k_median_test


def test_gb1021_basic():
    """Test basic functionality with documented k-sample input."""
    rng = np.random.default_rng(42)
    n_per = 50
    samples = [rng.normal(0.0, 1.0, n_per).tolist() for _ in range(3)]
    result = gibbons_k_median_test(samples)

    # Result must be a mapping with the documented keys.
    assert isinstance(result, dict)
    for key in ("statistic", "df", "p_value", "u", "t",
                "median", "prob", "k", "n", "method"):
        assert key in result

    # Independent recomputation of Q from the documented formula
    # Q = N^2 / (t (N - t)) * sum_i (u_i - n_i * t / N)^2 / n_i
    pooled = sorted(v for s in samples for v in s)
    nn = len(pooled)
    if nn % 2:
        d = pooled[nn // 2]
    else:
        d = (pooled[nn // 2 - 1] + pooled[nn // 2]) / 2.0
    u = [sum(1 for v in s if v < d) for s in samples]
    ns = [len(s) for s in samples]
    t = sum(u)
    assert t > 0 and t < nn, "combined median must split the pooled data"
    q_expected = (float(nn) ** 2 / (t * (nn - t))) * sum(
        (u[i] - ns[i] * t / float(nn)) ** 2 / ns[i] for i in range(len(samples))
    )

    assert result["k"] == len(samples)
    assert result["n"] == nn
    assert list(result["u"]) == u
    assert result["t"] == t
    assert result["median"] == float(d)
    assert result["df"] == len(samples) - 1
    assert result["statistic"] == q_expected
    assert result["prob"] > 0.0
    assert result["method"] == "k-sample median test (Sec. 10.2)"


def test_gb1021_edge():
    """Test edge case: two-sample input with a clear median split."""
    rng = np.random.default_rng(42)
    s1 = rng.normal(0.0, 1.0, 60).tolist()
    s2 = rng.normal(2.0, 1.0, 60).tolist()
    samples = [s1, s2]

    result = gibbons_k_median_test(samples)
    assert isinstance(result, dict)
    assert result["k"] == 2
    assert result["df"] == 1
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["statistic"] >= 0.0

    pooled = sorted(v for s in samples for v in s)
    nn = len(pooled)
    d = pooled[nn // 2] if nn % 2 else (pooled[nn // 2 - 1] + pooled[nn // 2]) / 2.0
    u = [sum(1 for v in s if v < d) for s in samples]
    ns = [len(s) for s in samples]
    t = sum(u)
    q_expected = (float(nn) ** 2 / (t * (nn - t))) * sum(
        (u[i] - ns[i] * t / float(nn)) ** 2 / ns[i] for i in range(len(samples))
    )
    assert result["statistic"] == q_expected
