"""Tests for chemsc.chemscore_dock."""

from morie.fn import _array_core as np

from morie.fn.chemsc import chemscore_dock


def _row(x, y, z, role, att=None):
    """Build an atom row; att=(ax,ay,az) or None."""
    if att is None:
        return [(x, y, z), role, None]
    return [(x, y, z), role, (att[0], att[1], att[2])]


def test_chemsc_basic():
    """Test basic functionality with a simple donor-acceptor pair."""
    # One receptor donor with its hydrogen position known,
    # one ligand acceptor with its attached heavy atom known.
    # Place atoms so distance d between heavy atoms is large -> no hbond term,
    # no lipophilic/lipo/metal contacts -> score comes only from general clashes.
    receptor = [
        [0.0, 0.0, 0.0, "donor", 0.0, 0.0, 1.0],   # donor heavy at origin, H along +z
    ]
    ligand = [
        [5.0, 0.0, 0.0, "acceptor", 4.0, 0.0, 0.0],  # acceptor heavy at (5,0,0)
    ]
    result = chemscore_dock(receptor, ligand)
    assert isinstance(result, dict)
    # The function returns a RichResult; document the expected key.
    assert "estimate" in result or "statistic" in result
    # When the only pair is donor/acceptor with valid partner coords,
    # a hydrogen bond term IS built (distance is 5.0).
    # The exact numeric estimate is not asserted; we only check it is finite.
    score = result.get("estimate", result.get("statistic"))
    import math
    assert math.isfinite(score)


def test_chemsc_edge():
    """Test edge case: empty inputs produce a degenerate (finite) score."""
    receptor = []
    ligand = []
    result = chemscore_dock(receptor, ligand)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result
