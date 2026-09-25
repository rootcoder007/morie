"""Tests for sscompv.survival_competing_validation."""

from morie.fn.sscompv import survival_competing_validation

# Five subjects: two early cause-1 events, one censoring, one competing
# event, one late cause-1 event.
TIME = [1.0, 2.0, 3.0, 4.0, 5.0]
EVENT = [1, 1, 0, 2, 1]


def test_sscompv_basic():
    """Wolbers concordance, counted by hand on a five-subject example.

    Comparable pairs (i has a cause-1 event, and T_i < T_j or j had a
    competing event): i=0 against every other subject (4 pairs), i=1
    against j=2,3,4 (3 pairs), and i=4 against the competing-event
    subject j=3 (1 pair) -- 8 comparable pairs. With monotonically
    decreasing risks the first 7 are concordant and the last is not.
    """
    pred = [0.9, 0.5, 0.4, 0.3, 0.2]
    result = survival_competing_validation(TIME, EVENT, pred)

    assert result["comparable"] == 8
    assert result["concordant"] == 7.0
    assert result["tied"] == 0.0
    assert abs(result["estimate"] - 7.0 / 8.0) < 1e-12
    assert result["method"] == "Wolbers et al (2014) competing-risks concordance"

    # Reversing every prediction turns each concordant pair discordant,
    # so with no ties the concordance is exactly 1 - C.
    flipped = survival_competing_validation(TIME, EVENT, [-p for p in pred])
    assert flipped["comparable"] == 8
    assert flipped["concordant"] == 1.0
    assert abs(flipped["estimate"] - 1.0 / 8.0) < 1e-12


def test_sscompv_ties_count_one_half():
    """A tied risk pair adds 1/2, not 1, to the numerator."""
    pred = [0.9, 0.5, 0.4, 0.3, 0.3]
    result = survival_competing_validation(TIME, EVENT, pred)
    assert result["comparable"] == 8
    assert result["concordant"] == 7.0
    assert result["tied"] == 1.0
    assert abs(result["estimate"] - 7.5 / 8.0) < 1e-12


def test_sscompv_edge():
    """Length mismatches and pair-free data are rejected."""
    try:
        survival_competing_validation(TIME, EVENT, [0.1, 0.2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unequal lengths")

    try:
        survival_competing_validation(TIME, [0, 0, 0, 0, 0], [0.1] * 5)
    except ValueError as exc:
        assert "comparable" in str(exc)
    else:
        raise AssertionError("expected ValueError when no pair is comparable")

    # A single cause-1 subject with a single competing-event partner gives
    # exactly one comparable pair, and the direction decides the estimate.
    one = survival_competing_validation([5.0, 1.0], [1, 2], [0.8, 0.2])
    assert one["comparable"] == 1
    assert one["concordant"] == 1.0
    assert one["estimate"] == 1.0
