"""Tests for rtwall (Wallinga-Teunis case reproduction numbers).

Replaces the generated stub, which imported ``rt_wallinga_teunis``.
"""

from morie.fn.rtwall import rtwall


def test_the_last_case_infects_nobody():
    onsets = [0, 1, 2, 3, 10]
    res = rtwall(onsets, [0.0, 0.5, 0.3, 0.2])
    assert abs(res["r_case"][-1]) < 1e-12


def test_the_attribution_conserves_infections():
    # every case that has an identifiable infector contributes exactly
    # one unit of infection, spread over its possible infectors, so
    # sum R_j minus that count is zero. The module reports the residual
    # as mass_check, and it is the identity worth testing.
    onsets = [0, 1, 1, 2, 3, 4]
    res = rtwall(onsets, [0.0, 0.6, 0.4])
    assert abs(res["mass_check"]) < 1e-9
    assert res["n_cases"] == 6
    # not every case is attributable: with support {1, 2} a case can only
    # be assigned to an onset one or two days earlier
    assert 0 < sum(res["r_case"]) <= len(onsets) - 1


def test_a_growing_epidemic_has_r_above_one_early():
    onsets = [0] + [1] * 2 + [2] * 4 + [3] * 8 + [4] * 16
    res = rtwall(onsets, [0.0, 1.0])
    assert res["r_case"][0] > 1.0


def test_an_unnormalised_generation_interval_is_rescaled():
    # weights of 2 and 2 mean the same thing as 0.5 and 0.5
    a = rtwall([0, 1, 2], [0.0, 2.0, 2.0])
    b = rtwall([0, 1, 2], [0.0, 0.5, 0.5])
    for i in range(3):
        assert abs(a["r_case"][i] - b["r_case"][i]) < 1e-12
    assert abs(a["mass_check"]) < 1e-9


def test_daily_and_case_numbers_are_both_reported():
    onsets = [0, 1, 1, 2, 3]
    res = rtwall(onsets, [0.0, 0.7, 0.3])
    assert len(res["r_case"]) == len(onsets)
    assert res["r_daily"]


def test_validation():
    for call in (lambda: rtwall([1], [0.0, 1.0]),
                 lambda: rtwall([1, 2], []),
                 lambda: rtwall([1, 2], [-0.5, 1.0])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
