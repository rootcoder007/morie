"""Tests for gb1241r.gibbons_concordance_rho_link."""

from morie.fn import _array_core as np

from morie.fn.gb1241r import gibbons_concordance_rho_link


def test_gb1241r_basic():
    """Test basic functionality against the documented formula."""
    W = 0.75
    k = 4
    result = gibbons_concordance_rho_link(W, k)

    # Documented return keys.
    assert "rho_av" in result
    assert "W" in result
    assert "k" in result
    assert "rho_min" in result
    assert "estimate" in result

    # Formula (12.4.6): rho_av = (k*W - 1) / (k - 1).
    expected_rho = (k * W - 1.0) / (k - 1.0)
    assert result["rho_av"] == expected_rho
    assert result["estimate"] == expected_rho
    assert result["W"] == float(W)
    assert result["k"] == k

    # Inverse (12.4.7) must reproduce W: W = (rho_av*(k-1) + 1)/k.
    expected_W_back = (expected_rho * (k - 1) + 1.0) / k
    assert result["inverse_check"] == expected_W_back


def test_gb1241r_edge():
    """Test boundary values and the documented -1/(k-1) floor."""
    k = 4

    # Perfect concordance: W=1 forces rho_av=1.
    r1 = gibbons_concordance_rho_link(1.0, k)
    assert r1["rho_av"] == 1.0
    assert r1["rho_min"] == -1.0 / (k - 1.0)

    # Bottom of the documented W range: W=0 gives rho_av = -1/(k-1).
    r0 = gibbons_concordance_rho_link(0.0, k)
    expected_rho0 = (k * 0.0 - 1.0) / (k - 1.0)
    assert r0["rho_av"] == expected_rho0
    assert round(r0["rho_av"], 6) == round(expected_rho0, 6)
