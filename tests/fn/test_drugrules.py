"""drugrules: each filter on its boundaries."""

from morie.fn.drugrules import egan_egg, lipinski_rule_of_five, oral_bioavailability_rules, veber_rules


def test_lipinski_boundaries():
    assert lipinski_rule_of_five(500, 5, 5, 10).violations == 0
    assert lipinski_rule_of_five(500.1, 5, 5, 10).passes  # one violation allowed
    assert not lipinski_rule_of_five(501, 5.1, 5, 10).passes
    assert not lipinski_rule_of_five(501, 5.1, 6, 11, max_violations=3).passes
    assert lipinski_rule_of_five(501, 5.1, 6, 10, max_violations=3).passes


def test_veber_egan_composite():
    assert veber_rules(10, psa=140).passes and not veber_rules(11, psa=100).passes
    assert veber_rules(9, hbond_total=12).passes and not veber_rules(9, hbond_total=13).passes
    assert egan_egg(131.6, -1.0).passes and not egan_egg(131.7, 2.0).passes and not egan_egg(50, 5.9).passes
    r = oral_bioavailability_rules(480, 5.5, 3, 8, 12, 150)
    assert (r.lipinski, r.veber, r.egan, r.n_passed) == (True, False, False, 1)
    try:
        veber_rules(5)
    except ValueError:
        pass
    else:
        raise AssertionError("missing polar criterion must raise")
