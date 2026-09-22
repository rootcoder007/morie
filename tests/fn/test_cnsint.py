"""Tests for cnsint.concurrent_calibration."""

from morie.fn import _array_core as np

from morie.fn.cnsint import concurrent_calibration


def _make_responses(rng, n, k, theta, b):
    """Build an n x k matrix of 0/1 responses from the Rasch model."""
    rows = []
    for i in range(n):
        row = []
        for j in range(k):
            p = 1.0 / (1.0 + float(np.exp(-(theta[i] - b[j]))))
            u = float(rng.random())
            row.append(1.0 if u < p else 0.0)
        rows.append(row)
    return rows


def test_cnsint_basic():
    """Test basic functionality with valid Rasch-shaped inputs."""
    rng = np.random.default_rng(43)
    n_f, n_r = 30, 30
    k = 5
    theta_f = list(rng.normal(0, 1, n_f))
    theta_r = list(rng.normal(0, 1, n_r))
    b_true = [float(v) for v in rng.normal(0, 1, k)]

    rows = _make_responses(rng, n_f, k, theta_f, b_true)
    rows += _make_responses(rng, n_r, k, theta_r, b_true)
    y = rows

    group = [0] * n_f + [1] * n_r
    anchor = [0, 2, 4]

    result = concurrent_calibration(y, group=group, anchor=anchor, iters=50)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "b_focal" in result
    assert "b_reference" in result
    assert "drift" in result
    assert "theta_mean_focal" in result
    assert "theta_mean_reference" in result

    # documented metadata
    assert result["n"] == n_f + n_r
    assert result["k"] == k
    assert result["n_anchor"] == len(anchor)

    # drift must have one entry per anchor item
    assert len(result["drift"]) == len(anchor)

    # estimate must equal the mean absolute drift (independent computation)
    drift = result["drift"]
    expected_estimate = sum(abs(v) for v in drift) / len(drift)
    assert abs(result["estimate"] - expected_estimate) < 1e-12


def test_cnsint_edge():
    """Single-group run: drift and theta means collapse correctly."""
    rng = np.random.default_rng(43)
    n = 40
    k = 4
    theta = list(rng.normal(0, 1, n))
    b_true = [float(v) for v in rng.normal(0, 1, k)]

    y = _make_responses(rng, n, k, theta, b_true)
    anchor = [1, 3]

    # group=None -> everyone is one group
    result = concurrent_calibration(y, group=None, anchor=anchor, iters=50)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == n
    assert result["k"] == k
    assert result["n_anchor"] == len(anchor)

    # With a single group the function documents drift = 0 for every anchor
    expected_drift = [0.0] * len(anchor)
    assert list(result["drift"]) == expected_drift
    assert result["estimate"] == 0.0
