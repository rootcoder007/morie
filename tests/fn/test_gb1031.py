"""Tests for gb1031.gibbons_k_ctrl_median."""

from morie.fn import _array_core as np

from morie.fn.gb1031 import gibbons_k_ctrl_median


def test_gb1031_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    groups = rng.normal(0, 1, 100)
    rng2 = np.random.default_rng(42)
    control = rng2.normal(0, 1, 100)
    result = gibbons_k_ctrl_median([groups, control])
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "counts" in result
    assert "cuts" in result
    assert "r" in result
    assert "pcell" in result
    assert "df" in result
    assert "k" in result
    assert "method" in result
    assert isinstance(result["statistic"], float)
    assert result["statistic"] >= 0.0
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["k"] == 2
    assert result["df"] == 1  # (k-1)*q with q=1 (default median)

    # Independent computation: with k=2, q=1, the table is 1 x 2.
    # pcell = [(r_1-0)/(n_1+1), (n_1+1-r_1)/(n_1+1)] with r_1 = floor(n1*0.5)+1.
    n1 = 100
    r1 = n1 * 0.5 + 1  # floor(50) + 1 = 51
    pcell = [(r1 - 0) / (n1 + 1.0), (n1 + 1 - r1) / (n1 + 1.0)]
    ctrl_sorted = sorted(control.tolist())
    cut = ctrl_sorted[int(r1) - 1]
    # Count observations in groups falling in each block.
    row = [0, 0]
    for v in groups.tolist():
        if v > cut:
            row[1] += 1
        else:
            row[0] += 1
    ni = row[0] + row[1]
    expected_stat = 0.0
    for j in range(2):
        e = ni * pcell[j]
        if e > 0.0:
            expected_stat += (row[j] - e) ** 2 / e
    # The function's statistic equals the independent computation.
    assert result["statistic"] == expected_stat

    # The cuts and r match the independent computation.
    assert list(result["r"]) == [int(r1)]
    assert list(result["cuts"]) == [float(cut)]
    # pcell should be the Beta-mean cell probabilities.
    assert list(result["pcell"]) == pcell
    # The single treatment row should match our independently built row.
    assert list(result["counts"][0]) == row


def test_gb1031_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    groups = rng.normal(0, 1, 100)
    rng2 = np.random.default_rng(42)
    control = rng2.normal(0, 1, 100)
    result = gibbons_k_ctrl_median([groups, control])
    assert isinstance(result, dict)
    assert "statistic" in result
    assert result["k"] == 2
