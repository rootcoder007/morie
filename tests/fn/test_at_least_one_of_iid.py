"""Tests for morie.fn.at_least_one_of_iid: values recomputed from first principles."""

from morie.fn.at_least_one_of_iid import at_least_one_of_iid


def test_complement_form():
    for p, k in ((1 / 6, 3), (0.3, 5), (0.9, 1)):
        r = at_least_one_of_iid(p, k)
        assert abs(r["p_at_least_one"] - (1 - (1 - p) ** k)) < 1e-14
    # book dice example: 3p - 3p^2 + p^3
    p = 1 / 6
    assert abs(at_least_one_of_iid(p)["p_at_least_one"] - (3 * p - 3 * p * p + p**3)) < 1e-15
