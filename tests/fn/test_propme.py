"""Tests for propme (proportion mediated).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.propme import propme


def test_the_closed_form():
    a, b, cp = 0.5, 0.4, 0.3
    res = propme(a, b, cp)
    ab = a * b
    assert abs(res["indirect"] - ab) < 1e-12
    assert abs(res["total"] - (cp + ab)) < 1e-12
    assert abs(res["estimate"] - ab / (cp + ab)) < 1e-12
    assert abs(res["ratio"] - ab / cp) < 1e-12


def test_no_direct_effect_means_everything_is_mediated():
    res = propme(0.5, 0.4, 0.0)
    assert abs(res["estimate"] - 1.0) < 1e-12


def test_no_indirect_effect_means_nothing_is_mediated():
    res = propme(0.0, 0.4, 0.3)
    assert abs(res["estimate"]) < 1e-12
    assert abs(res["ratio"]) < 1e-12


def test_opposite_signs_are_flagged_because_the_proportion_misleads():
    # inconsistent mediation: the indirect and direct effects cancel, so
    # a "proportion mediated" above 1 or below 0 is not a proportion
    res = propme(0.5, 0.4, -0.1)
    assert not res["same_sign"]
    assert res["estimate"] > 1.0


def test_consistent_mediation_is_flagged_as_such():
    assert propme(0.5, 0.4, 0.3)["same_sign"]


def test_a_zero_total_effect_is_refused():
    try:
        propme(0.5, 0.4, -0.2)      # c' + ab = 0 exactly
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
