"""Tests for dueldqn. Full anchor: ledger/wave3/anchor_ts_family.py."""
import pytest
from morie.fn import _s03core as k
from morie.fn.dueldqn import (double_q_target, dueling_aggregate,
                              dueling_step)

V, A = 3.0, [1.0, -2.0, 0.5]


@pytest.mark.parametrize("mode", ["mean", "max"])
def test_the_correction_makes_the_decomposition_identified(mode):
    """Shifting the whole advantage stream must leave Q EXACTLY
    unchanged -- that is what identifiability means here."""
    q0 = dueling_aggregate(V, A, mode=mode)
    q1 = dueling_aggregate(V, [a + 7.0 for a in A], mode=mode)
    assert q0 == pytest.approx(q1, abs=1e-14)


def test_the_naive_form_is_unidentifiable():
    q0 = dueling_aggregate(V, A, mode="naive")
    q1 = dueling_aggregate(V, [a + 7.0 for a in A], mode="naive")
    assert max(abs(q0[i] - q1[i]) for i in range(3)) > 6.9
    # and it IS invariant to the opposite shift, which is the problem
    q2 = dueling_aggregate(V + 5.0, [a - 5.0 for a in A], mode="naive")
    assert q2 == pytest.approx(q0, abs=1e-14)


def test_the_two_aggregations_have_their_stated_meanings():
    assert max(dueling_aggregate(V, A, mode="max")) == pytest.approx(
        V, abs=1e-14)
    assert k.mean(dueling_aggregate(V, A, mode="mean")) == \
        pytest.approx(V, abs=1e-14)


def test_the_double_q_target_splits_selection_from_valuation():
    got = double_q_target(1.0, 0.9, [0.0, 5.0, 1.0], [7.0, 2.0, 3.0])
    assert got == pytest.approx(1.0 + 0.9 * 2.0, abs=1e-14)
    assert double_q_target(1.5, 0.9, [1.0], [9.0], done=True) == 1.5


def test_argument_checks():
    with pytest.raises(ValueError):
        dueling_aggregate(V, A, mode="nope")
    with pytest.raises(ValueError):
        dueling_aggregate(V, [])
    with pytest.raises(ValueError):
        dueling_step(V, A, 9, 1.0, 0.9, V, A, V, A)
    with pytest.raises(ValueError):
        double_q_target(1.0, 0.9, [1.0, 2.0], [1.0])
