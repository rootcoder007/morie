"""Tests for epbias.exposure_misclass_bias."""

from morie.fn import _array_core as np

from morie.fn.epbias import exposure_misclass_bias


def test_epbias_basic():
    """Test basic functionality."""
    A_obs = [40, 10]
    N = [100, 100]
    Se = 0.9
    Sp = 0.95
    result = exposure_misclass_bias(A_obs, Se, Sp, N=N)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "a_true" in result
    assert "a_obs" in result
    assert "totals" in result
    assert "prevalence" in result
    assert "or_obs" in result
    assert "or_true" in result
    assert "sensitivity" in result
    assert "specificity" in result
    assert "n" in result
    assert "method" in result

    # Independent computation of the formula: A = (A_obs - (1 - Sp) * N) / (Se + Sp - 1)
    expected_a_true_0 = (A_obs[0] - (1.0 - Sp) * N[0]) / (Se + Sp - 1.0)
    expected_a_true_1 = (A_obs[1] - (1.0 - Sp) * N[1]) / (Se + Sp - 1.0)
    assert result["a_true"][0] == expected_a_true_0
    assert result["a_true"][1] == expected_a_true_1
    assert result["estimate"] == expected_a_true_0
    assert result["a_obs"] == [40.0, 10.0]
    assert result["totals"] == [100.0, 100.0]
    assert result["prevalence"][0] == expected_a_true_0 / N[0]
    assert result["prevalence"][1] == expected_a_true_1 / N[1]
    assert result["sensitivity"] == [0.9, 0.9]
    assert result["specificity"] == [0.95, 0.95]
    assert result["n"] == 2

    # Observed OR: (40/10) / ((100-40)/(100-10))
    expected_or_obs = (A_obs[0] * (N[1] - A_obs[1])) / (A_obs[1] * (N[0] - A_obs[0]))
    assert result["or_obs"] == expected_or_obs

    # Corrected OR computed independently from a_true values
    expected_or_true = (expected_a_true_0 * (N[1] - expected_a_true_1)) / (
        expected_a_true_1 * (N[0] - expected_a_true_0)
    )
    assert result["or_true"] == expected_or_true


def test_epbias_edge():
    """Test edge cases: single-group two-element input read as (exposed, unexposed)."""
    # When N is omitted, A_obs must be length 2: (exposed, unexposed) of a single group.
    A_obs = [40, 60]
    Se = 0.9
    Sp = 0.95
    result = exposure_misclass_bias(A_obs, Se, Sp)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["n"] == 1
    assert result["totals"] == [100.0]
    # Independent computation
    expected_a_true = (A_obs[0] - (1.0 - Sp) * 100.0) / (Se + Sp - 1.0)
    assert result["a_true"][0] == expected_a_true
    assert result["estimate"] == expected_a_true
    # Only one group => odds ratios are undefined
    assert result["or_obs"] != result["or_obs"]  # NaN check
    assert result["or_true"] != result["or_true"]  # NaN check
