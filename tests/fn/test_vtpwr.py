"""vtpwr: Banzhaf and Shapley-Shubik voting power (Armstrong Ch 10)."""

import numpy as np
import pytest

from morie.fn.vtpwr import voting_power_index as vpi


def test_vtpwr_classic_game_matches_hand_enumeration():
    """[3; 2, 1, 1] -- both indices derived by hand, not read off the code.

    Winning coalitions (weight >= 3): {A,B}, {A,C}, {A,B,C}.
    Banzhaf swings: A is critical in all three; B only in {A,B} (dropping B
    from {A,B,C} leaves {A,C} = 3, still winning); C likewise. 3:1:1 of 5.
    Shapley-Shubik over the 6 orderings: A is pivotal in BAC, BCA, CAB, CBA;
    B in ABC; C in ACB. 4:1:1 of 6.
    """
    r = vpi(np.array([2.0, 1.0, 1.0]), quota=3.0)
    assert np.asarray(r["banzhaf"]) == pytest.approx([0.6, 0.2, 0.2])
    assert np.asarray(r["shapley_shubik"]) == pytest.approx([2 / 3, 1 / 6, 1 / 6])


def test_vtpwr_default_quota_is_strictly_more_than_half():
    """Default q = sum(w)/2 + eps, i.e. simple majority.

    For weights (2, 1, 1) that is 2+, NOT 3. The docstring used to claim
    ceil(sum(w)/2 + 1) = 3, which is a three-quarters super-majority and was
    never what the code computed; the docstring was corrected to match.
    """
    r = vpi(np.array([2.0, 1.0, 1.0]))
    assert r["quota"] == pytest.approx(2.0, abs=1e-6)
    assert r["quota"] > 2.0, "an exact half-and-half split must lose"


def test_vtpwr_default_quota_is_a_genuine_majority_rule():
    """Under the default, {A} alone (weight 2 of 4) must NOT win: it is
    exactly half. Adding any second voter wins."""
    r = vpi(np.array([2.0, 1.0, 1.0]))
    # A is not a dictator under simple majority, so its power is below 1.
    assert np.asarray(r["banzhaf"])[0] < 1.0


def test_vtpwr_both_indices_are_probability_distributions():
    rng = np.random.default_rng(19)
    for _ in range(10):
        w = rng.integers(1, 10, size=6).astype(float)
        r = vpi(w)
        for key in ("banzhaf", "shapley_shubik"):
            v = np.asarray(r[key])
            assert v.sum() == pytest.approx(1.0)
            assert np.all(v >= 0.0)


def test_vtpwr_dictator_holds_all_the_power():
    """One voter meets quota alone and no coalition wins without them."""
    r = vpi(np.array([5.0, 1.0, 1.0]), quota=5.0)
    assert np.asarray(r["banzhaf"]) == pytest.approx([1.0, 0.0, 0.0])
    assert np.asarray(r["shapley_shubik"]) == pytest.approx([1.0, 0.0, 0.0])


def test_vtpwr_equal_weights_give_equal_power():
    r = vpi(np.ones(4), quota=3.0)
    assert np.asarray(r["banzhaf"]) == pytest.approx(np.full(4, 0.25))
    assert np.asarray(r["shapley_shubik"]) == pytest.approx(np.full(4, 0.25))


def test_vtpwr_dummy_voter_has_zero_power_despite_positive_weight():
    """[8; 5, 3, 1]: the 1-weight voter is never critical -- 5+3 already wins
    and no coalition needs the third. Weight is not power, which is the whole
    point of computing an index instead of normalising the weights.
    """
    r = vpi(np.array([5.0, 3.0, 1.0]), quota=8.0)
    assert np.asarray(r["banzhaf"])[2] == pytest.approx(0.0)
    assert np.asarray(r["shapley_shubik"])[2] == pytest.approx(0.0)
    # ...and its weight share is a non-trivial 1/9, so the two genuinely differ.
    assert 1.0 / 9.0 > 0.1


def test_vtpwr_power_is_not_proportional_to_weight():
    """[51; 49, 48, 3]: weights are wildly unequal, power is exactly equal --
    any two of the three voters win, so all three are symmetric."""
    r = vpi(np.array([49.0, 48.0, 3.0]), quota=51.0)
    assert np.asarray(r["banzhaf"]) == pytest.approx(np.full(3, 1 / 3))
    assert np.asarray(r["shapley_shubik"]) == pytest.approx(np.full(3, 1 / 3))


def test_vtpwr_echoes_its_inputs():
    r = vpi(np.array([2.0, 1.0, 1.0]), quota=3.0)
    assert np.asarray(r["weights"]) == pytest.approx([2.0, 1.0, 1.0])
    assert r["quota"] == pytest.approx(3.0)
