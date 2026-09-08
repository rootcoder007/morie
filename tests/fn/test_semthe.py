"""Tests for semthe.sem_theta (IRT standard error)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.semthe import sem_theta


def test_semthe_matches_the_hand_computed_single_item():
    """One item with a = 1, b = 0 at theta = 0: P = 1/2, so
    I = 1 * 1/4 and SE = 2 exactly (Lord 1980, Ch. 5)."""
    r = sem_theta(0.0, [[1.0, 0.0]])
    assert float(r["information"]) == pytest.approx(0.25, abs=1e-12)
    assert float(r["se"]) == pytest.approx(2.0, abs=1e-12)


def test_semthe_information_adds_across_items():
    """Two identical items double the information, dividing the SE by
    sqrt(2) -- additivity is the defining property."""
    one = float(sem_theta(0.3, [[1.2, 0.0]])["information"])
    two = float(sem_theta(0.3, [[1.2, 0.0], [1.2, 0.0]])["information"])
    assert two == pytest.approx(2 * one, rel=1e-12)


def test_semthe_se_is_smallest_where_items_match_ability():
    """Items at b = 0: the SE at theta = 0 must beat the SE at theta = 3.
    This is the fact adaptive testing exploits."""
    items = [[1.5, 0.0]] * 10
    se0 = float(sem_theta(0.0, items)["se"])
    se3 = float(sem_theta(3.0, items)["se"])
    assert se0 < se3 / 2


def test_semthe_vector_theta_and_validation():
    r = sem_theta([-1.0, 0.0, 1.0], [[1.0, 0.0], [2.0, 1.0]])
    assert np.asarray(r["se"]).shape == (3,)
    with pytest.raises(ValueError, match=r"\(k, 2\)"):
        sem_theta(0.0, [1.0, 2.0, 3.0])
