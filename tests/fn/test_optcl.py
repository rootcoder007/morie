"""optcl: Optimal Classification cutting point.

Poole, K. T. (2000). "Nonparametric unfolding of binary choice data."
*Political Analysis*, 8(3), 211-237 -- in the library, verified from the PDF.
Also Armstrong et al., section 5.4 "Nonparametric Methods - Optimal
Classification", printed p.156 (NOT a Ch 7-10; that book has six chapters).
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.optcl import optimal_classification as oc


def test_optcl_finds_a_perfectly_separating_cut():
    """Ideal points below 0 vote nay, above vote yea: a cut near 0 classifies
    every legislator correctly."""
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    votes = np.array([0, 0, 0, 1, 1, 1])
    r = oc(x, votes)
    assert r["correct_class"] == 6
    assert -1.0 <= r["cut"] <= 1.0


def test_optcl_reports_the_polarity_of_the_vote():
    """Reversing which side votes yea must flip the polarity, not the fit --
    OC has to search both orientations or it misclassifies half the votes."""
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    a = oc(x, np.array([0, 0, 0, 1, 1, 1]))
    b = oc(x, np.array([1, 1, 1, 0, 0, 0]))
    assert a["correct_class"] == b["correct_class"] == 6
    assert a["polarity"] != b["polarity"]


def test_optcl_pre_is_proportional_reduction_in_error():
    """PRE compares OC's errors with always guessing the modal outcome. A
    perfect cut on a balanced vote gives PRE = 1."""
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    r = oc(x, np.array([0, 0, 0, 1, 1, 1]))
    assert r["pre"] == pytest.approx(1.0)


def test_optcl_a_lopsided_vote_carries_little_information():
    """When almost everyone votes the same way the modal guess is already
    nearly perfect, so PRE is small even though accuracy is high."""
    x = np.linspace(-3, 3, 20)
    votes = np.ones(20, dtype=int); votes[0] = 0
    r = oc(x, votes)
    assert r["correct_class"] >= 19
    assert r["pre"] <= 1.0


def test_optcl_cannot_beat_the_data_on_a_scrambled_vote():
    """With votes unrelated to the ideal points, classification must be far
    from perfect -- otherwise the cut is overfitting."""
    rng = np.random.default_rng(3301)
    x = np.sort(rng.standard_normal(60))
    votes = rng.integers(0, 2, 60)
    r = oc(x, votes)
    assert r["correct_class"] < 60


def test_optcl_correct_class_never_exceeds_n():
    rng = np.random.default_rng(3307)
    for n in (10, 40, 100):
        x = np.sort(rng.standard_normal(n))
        r = oc(x, rng.integers(0, 2, n))
        assert 0 <= r["correct_class"] <= n
        assert r["n"] == n


def test_optcl_is_invariant_to_a_monotone_rescaling_of_the_scale():
    """OC is nonparametric -- it uses only the ORDER of the ideal points, so
    an affine rescaling cannot change how many votes it classifies."""
    rng = np.random.default_rng(3313)
    x = np.sort(rng.standard_normal(40))
    votes = (x > 0.2).astype(int)
    assert oc(x, votes)["correct_class"] == oc(5 * x + 3, votes)["correct_class"]
