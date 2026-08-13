"""Tests for frfgrf. Full anchor: ledger/wave3/anchor_grf_family.py."""
import math
import pytest
from morie.fn.frfgrf import beta_min, forest_fit_check
from ._grf_fixture import confounded


@pytest.fixture(scope="module")
def d():
    return confounded(400, 21)


def test_beta_min_matches_its_closed_form():
    want = 1.0 - (1.0 + (3.0 / 0.5)
                  * (math.log(20.0) / math.log(1.0 / 0.95))) ** -1.0
    assert beta_min(3, 0.05, 0.5) == pytest.approx(want, abs=1e-12)
    assert beta_min(10, 0.05, 0.5) > beta_min(3, 0.05, 0.5)
    with pytest.raises(ValueError):
        beta_min(3, 0.7, 0.5)
    with pytest.raises(ValueError):
        beta_min(3, 0.05, 0.0)


def test_an_honest_forest_passes_and_an_adaptive_one_does_not(d):
    """A diagnostic that cleared a non-honest forest would be worse than
    no diagnostic, so honesty is tested structurally rather than read
    off the constructor's arguments."""
    ok = forest_fit_check(d["y"], d["X"], n_trees=60, min_leaf=5,
                          seed=8)
    assert ok["passes"]
    assert ok["honesty"]["splits_stable_under_I_permutation"]
    assert ok["honesty"]["splits_move_under_J_permutation"]
    bad = forest_fit_check(d["y"], d["X"], n_trees=60, min_leaf=5,
                           seed=8, kind="adaptive")
    assert not bad["checks"]["honest"]
    assert not bad["passes"]


def test_argument_checks(d):
    with pytest.raises(ValueError):
        forest_fit_check(d["y"][:-1], d["X"])
    with pytest.raises(ValueError):
        forest_fit_check(d["y"][:20], d["X"][:20])
