"""Tests for gh_c12_7.ghosal_strict_sbvm."""

from morie.fn import _array_core as np

from morie.fn.gh_c12_7 import ghosal_strict_sbvm


def test_gh_c12_7_basic():
    """Test basic functionality: all three conditions satisfied."""
    result = ghosal_strict_sbvm(
        prior_mass_ok=True, lan_remainder=0.01, change_of_measure_gap=0.02, tol=0.05
    )
    # Documented return keys
    assert "estimate" in result
    assert "bvm_holds" in result
    assert "conditions" in result
    assert "method" in result

    # Compute the expected estimate independently from the formula:
    #   score = (0.0 if not prior_mass_ok else 1.0) - lan_remainder - change_of_measure_gap
    pm, lr, cg, tol = True, 0.01, 0.02, 0.05
    expected_score = (0.0 if not pm else 1.0) - lr - cg
    assert np.isclose(float(result["estimate"]), expected_score)

    # All three conditions satisfied -> overall flag is True
    assert result["bvm_holds"] is True
    assert list(result["conditions"]) == [True, True, True]

    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c12_7_failing_conditions():
    """When any documented condition fails, the aggregate flag is False."""
    # prior_mass_ok=False overrides everything
    result = ghosal_strict_sbvm(
        prior_mass_ok=False, lan_remainder=0.0, change_of_measure_gap=0.0, tol=0.05
    )
    assert result["bvm_holds"] is False
    assert result["conditions"][0] is False
    # score = 0.0 - 0.0 - 0.0
    assert np.isclose(float(result["estimate"]), 0.0)

    # LAN remainder exceeds tolerance
    result = ghosal_strict_sbvm(
        prior_mass_ok=True, lan_remainder=0.1, change_of_measure_gap=0.0, tol=0.05
    )
    assert result["bvm_holds"] is False
    assert result["conditions"][1] is False
    expected = 1.0 - 0.1 - 0.0
    assert np.isclose(float(result["estimate"]), expected)
