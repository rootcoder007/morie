"""Tests for alfvio.alphafold_violation."""

from morie.fn.alfvio import alphafold_violation


def test_alfvio_basic():
    """Test basic functionality."""
    result = alphafold_violation()
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_alfvio_edge():
    """Test edge cases."""
    result = alphafold_violation()
    assert isinstance(result, dict)


def test_alfvio_flat_bottom_penalties_recomputed():
    """Eqs (44)-(46): flat-bottom L1 beyond 12 sigma for bonds and angles
    (averaged), one-sided clash below d_lit - 1.5 (summed)."""
    import pytest

    blen, blit, bsig = [1.5, 1.2, 1.33], [1.33, 1.33, 1.33], [0.01, 0.01, 0.01]
    cang, clit, csig = [0.2, -0.5], [-0.3, -0.35], [0.02, 0.02]
    dnb, dlit = [2.0, 3.6], [4.0, 4.0]
    lb = sum(max(abs(a - b) - 12 * s, 0.0) for a, b, s in zip(blen, blit, bsig)) / 3
    la = sum(max(abs(a - b) - 12 * s, 0.0) for a, b, s in zip(cang, clit, csig)) / 2
    lc = sum(max(d - 1.5 - x, 0.0) for x, d in zip(dnb, dlit))
    r = alphafold_violation(blen, blit, bsig, cang, clit, csig, dnb, dlit)
    assert (r["bondlength"], r["bondangle"], r["clash"]) == pytest.approx((lb, la, lc), rel=1e-14)
    assert r["estimate"] == pytest.approx(lb + la + lc, rel=1e-14)
