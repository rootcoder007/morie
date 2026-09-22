"""Tests for frgrow.fragment_growing."""

from morie.fn import _array_core as np

from morie.fn.frgrow import fragment_growing


def test_frgrow_basic():
    """Test basic functionality."""
    # Parent fragment: (kd, hac, logp, mw, psa). Heavy-atom count must be
    # positive; last three entries are optional.
    fragment = (1.0e-6, 18, 1.5, 250.0, 60.0)
    # Each grown analogue shares the same tuple shape; name is optional.
    linker_lib = [
        (5.0e-7, 22, 2.1, 310.0, 75.0, "a1"),
        (2.0e-6, 20, 1.8, 280.0, 65.0, "a2"),
        (1.0e-7, 24, 2.4, 340.0, 80.0, "a3"),
    ]
    result = fragment_growing(fragment, linker_lib)

    # The function returns a dict with a fixed set of documented keys.
    assert isinstance(result, dict)
    assert "estimate" in result

    # Three analogues -> three per-row vectors and the right counts.
    assert len(result["kd"]) == 3
    assert len(result["hac"]) == 3
    assert len(result["d_hac"]) == 3
    assert len(result["group_efficiency"]) == 3
    assert result["n"] == 3

    # Atom-weighted blend of parent LE and group efficiency must equal
    # the analogue's own ligand efficiency (the identity the decision
    # rule rests on).
    parent_le = result["parent_le"]
    for le, ge, d_hac, hac, le_from_blend in zip(
            result["le"],
            result["group_efficiency"],
            result["d_hac"],
            result["hac"],
            result["le_from_blend"]):
        expected_blend = (parent_le * (hac - d_hac) + ge * d_hac) / hac
        assert le == expected_blend
        assert le_from_blend == expected_blend

    # 'improved' lists indices whose group efficiency exceeds parent LE.
    parent_hac = fragment[1]
    expected_improved = [
        i for i, (ge, hac) in enumerate(
            zip(result["group_efficiency"], result["hac"]))
        if ge > result["parent_le"]
    ]
    assert result["improved"] == expected_improved
    assert result["n_improved"] == len(expected_improved)

    # Ranking is by descending group efficiency, tie-broken by index.
    ge = result["group_efficiency"]
    expected_ranking = sorted(range(len(ge)), key=lambda i: (-ge[i], i))
    assert result["ranking"] == expected_ranking
    assert result["best"] == expected_ranking[0]
    assert result["estimate"] == ge[expected_ranking[0]]


def test_frgrow_edge():
    """Test edge cases: trailing optional fields may be absent or None."""
    # Same shape as the basic case, but logp/mw/psa omitted entirely.
    fragment = (1.0e-6, 18)
    linker_lib = [
        (5.0e-7, 22, None, None, None, "x"),
        (2.0e-6, 20, None, None, None, "y"),
    ]
    result = fragment_growing(fragment, linker_lib)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert len(result["kd"]) == 2
    assert len(result["hac"]) == 2
    # d_hac is the atom-cost of the addition.
    assert result["d_hac"] == [22 - 18, 20 - 18]
    assert result["n"] == 2
