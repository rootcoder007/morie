"""brdgr: bridge observations across sessions (Bailey 2007; Armstrong Ch 6)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.brdgr import bridge_observations as bo


def test_brdgr_two_id_lists_intersect_exactly():
    """Bridges are the members appearing in BOTH sessions: {2, 3} of 3 and 3."""
    r = bo([1, 2, 3], [2, 3, 4])
    assert r["n_bridges"] == 2
    assert sorted(np.asarray(r["bridge_ids"]).tolist()) == [2, 3]
    assert r["n1"] == 3 and r["n2"] == 3


def test_brdgr_share_is_relative_to_the_sessions():
    """2 bridges out of 3 members is a share of 2/3, not 2/6 or 2/4."""
    r = bo([1, 2, 3], [2, 3, 4])
    assert r["share"] == pytest.approx(2 / 3)


def test_brdgr_disjoint_sessions_have_no_bridges():
    """No overlap means nothing links the two scales -- the case that makes
    cross-chamber comparison impossible, so it must report 0, not fail."""
    r = bo([1, 2, 3], [7, 8, 9])
    assert r["n_bridges"] == 0
    assert r["share"] == pytest.approx(0.0)
    assert len(np.asarray(r["bridge_ids"])) == 0


def test_brdgr_identical_sessions_are_all_bridges():
    r = bo([1, 2, 3, 4], [1, 2, 3, 4])
    assert r["n_bridges"] == 4
    assert r["share"] == pytest.approx(1.0)


def test_brdgr_ignores_duplicates_and_order():
    """Membership is a set property: repeating an ID does not add a bridge."""
    a = bo([3, 1, 2, 2], [2, 3, 3, 4])
    b = bo([1, 2, 3], [2, 3, 4])
    assert a["n_bridges"] == b["n_bridges"] == 2


def test_brdgr_single_argument_mode_counts_nonzero_entries():
    """One-arg shortcut: x is a mask where a nonzero entry marks a bridge."""
    r = bo(np.array([1.0, 0.0, 2.0, 0.0, 5.0]))
    assert r["n_bridges"] == 3
