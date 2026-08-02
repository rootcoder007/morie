"""rcall: roll-call matrix summary, Poole-Rosenthal coding (Armstrong)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.rcall import roll_call_analysis as rc


def test_rcall_counts_a_binary_matrix_exactly():
    """3 legislators x 4 votes, hand-counted: 6 yea, 4 nay, 2 absent."""
    V = np.array(
        [
            [1.0, 1.0, 0.0, np.nan],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, np.nan, 0.0, 1.0],
        ]
    )
    r = rc(V)
    assert r["n"] == 3 and r["m"] == 4
    assert r["n_yea"] == 6
    assert r["n_nay"] == 4
    assert r["n_abs"] == 2


def test_rcall_applies_the_poole_rosenthal_decoding():
    """Codes 1/2/3 are yea, 4/5/6 are nay, 0/7/8/9 absent.

    Passing raw Poole-Rosenthal codes must give the same counts as the
    already-binary equivalent -- that mapping is the function's whole reason
    for accepting more than {0, 1, NaN}.
    """
    raw = np.array([[1.0, 2.0, 4.0, 9.0], [3.0, 5.0, 6.0, 1.0]])
    binary = np.array([[1.0, 1.0, 0.0, np.nan], [1.0, 0.0, 0.0, 1.0]])
    a, b = rc(raw), rc(binary)
    assert (a["n_yea"], a["n_nay"], a["n_abs"]) == (b["n_yea"], b["n_nay"], b["n_abs"])
    assert a["n_yea"] == 4 and a["n_nay"] == 3 and a["n_abs"] == 1


def test_rcall_marginals_are_per_vote_not_per_legislator():
    """marginal_yea has one entry per roll call (column), not per member."""
    V = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    r = rc(V)
    assert np.asarray(r["marginal_yea"]).shape == (2,)
    assert np.asarray(r["marginal_yea"]) == pytest.approx([3, 1])
    assert np.asarray(r["marginal_nay"]) == pytest.approx([0, 2])


def test_rcall_pct_yea_is_the_column_share():
    V = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert np.asarray(rc(V)["pct_yea"]) == pytest.approx([0.75, 0.5])


def test_rcall_lopsided_share_counts_the_near_unanimous_votes():
    """A vote where almost everyone agrees carries no spatial information --
    identifying those is why the statistic exists."""
    unanimous = np.ones((10, 1))
    split = np.array([[1.0]] * 5 + [[0.0]] * 5)
    assert rc(unanimous)["lopsided_pct"] > rc(split)["lopsided_pct"]


def test_rcall_absences_do_not_count_as_nays():
    """The distinction the encoding exists to preserve: NaN is not 0."""
    with_abs = rc(np.array([[1.0, np.nan]]))
    with_nay = rc(np.array([[1.0, 0.0]]))
    assert with_abs["n_abs"] == 1 and with_abs["n_nay"] == 0
    assert with_nay["n_abs"] == 0 and with_nay["n_nay"] == 1


def test_rcall_accepts_a_single_column_vector():
    r = rc(np.array([1.0, 0.0, 1.0]))
    assert r["n"] == 3 and r["m"] == 1
    assert r["n_yea"] == 2 and r["n_nay"] == 1
