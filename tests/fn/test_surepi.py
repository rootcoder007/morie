"""surepi -- EARS. Source: Hutwagner, L., Thompson, W., Seeman, G. M.
& Treadwell, T. (2003) Journal of Urban Health 80(2 Suppl 1), i89-i96
(the article prints no DOI)."""
import pytest

from morie.fn.surepi import (c1_mild, c2_medium, c3_ultra,
                             compound_smoothing, ears_detect,
                             salmonella_cusum)

FLAT = [10.0] * 20
SPIKE = FLAT + [40.0] + [10.0] * 5


def test_c1_baseline_is_seven_days_ending_one_day_back():
    r = c1_mild(SPIKE)
    assert (r["baseline_lag"], r["baseline_width"]) == (1, 7)


def test_c2_baseline_is_seven_days_ending_three_days_back():
    r = c2_medium(SPIKE)
    assert (r["baseline_lag"], r["baseline_width"]) == (3, 7)


def test_c1_on_a_flat_baseline_is_the_scaled_deviation():
    r = c1_mild(SPIKE, sigma_floor=1.0)
    assert r["statistic"][20] == pytest.approx(30.0, abs=1e-12)


def test_the_sigma_floor_scales_the_statistic():
    a = c1_mild(SPIKE, sigma_floor=1.0)["statistic"][20]
    b = c1_mild(SPIKE, sigma_floor=2.0)["statistic"][20]
    assert b == pytest.approx(a / 2.0, abs=1e-12)


def test_the_spike_is_flagged_and_the_flat_days_are_not():
    r = c1_mild(SPIKE)
    assert r["flag"][20] is True
    assert not any(r["flag"][t] for t in range(7, 20))


def test_days_without_a_full_baseline_are_none():
    r = c1_mild(SPIKE)
    assert r["statistic"][0] is None
    assert r["statistic"][6] is None
    assert r["statistic"][7] is not None


def test_c3_is_the_sum_of_three_consecutive_c2_values():
    mild = [10.0] * 15 + [10.0, 10.0, 14.0, 14.0, 14.0]
    c2 = c2_medium(mild)["statistic"]
    c3 = c3_ultra(mild)["statistic"]
    assert c3[-1] == pytest.approx(sum(c2[-3:]), abs=1e-12)


def test_c2_beats_c1_on_a_rising_ramp():
    ramp = [10.0] * 15 + [14.0, 18.0, 22.0, 26.0, 30.0]
    assert (c2_medium(ramp)["statistic"][-1]
            > c1_mild(ramp)["statistic"][-1])


def test_a_higher_threshold_flags_no_more_days():
    a = ears_detect(SPIKE, threshold=3.0)["n_flagged"]
    b = ears_detect(SPIKE, threshold=10.0)["n_flagged"]
    assert b <= a


def test_salmonella_cusum_matches_the_recursion():
    got = salmonella_cusum([3.0, 4.0, 12.0], mu0=4.0, sigma=2.0,
                           k_shift=1.0)["cusum"]
    S, want = 0.0, []
    for x in (3.0, 4.0, 12.0):
        S = max(0.0, S + (x - (4.0 + 2.0)) / 2.0)
        want.append(S)
    assert got == pytest.approx(want, abs=1e-13)


def test_salmonella_cusum_never_goes_negative():
    got = salmonella_cusum([0.0] * 10, mu0=50.0, sigma=2.0)["cusum"]
    assert all(v >= 0.0 for v in got)


def test_a_small_count_is_not_flagged_however_high_the_cusum():
    r = salmonella_cusum([3.0] * 5 + [4.0], mu0=0.0, sigma=1.0,
                         k_shift=0.0, min_count=5)
    assert r["cusum"][-1] > 0.5
    assert r["flag"][-1] is False


def test_compound_smoothing_flags_a_far_excursion():
    assert compound_smoothing([10.0] * 30, current=50.0)["flag"]


def test_compound_smoothing_does_not_flag_the_baseline():
    assert not compound_smoothing([10.0] * 30, current=10.0)["flag"]


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        ears_detect(SPIKE, method="C9")


def test_negative_counts_are_refused():
    with pytest.raises(ValueError):
        ears_detect([-1.0] * 30)


def test_a_series_shorter_than_the_baseline_is_refused():
    with pytest.raises(ValueError):
        ears_detect([1.0] * 6, method="C2")


def test_a_non_positive_sigma_floor_is_refused():
    with pytest.raises(ValueError):
        ears_detect(SPIKE, sigma_floor=0.0)


def test_a_non_positive_sigma_in_the_cusum_is_refused():
    with pytest.raises(ValueError):
        salmonella_cusum([1.0, 2.0], mu0=1.0, sigma=0.0)
