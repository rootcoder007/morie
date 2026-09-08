"""qrntcq -- quarantine efficacy. Source: Ashcroft, P., Lehtinen, S.,
Angst, D. C., Low, N. & Bonhoeffer, S. (2021) eLife 10, e63704,
doi:10.7554/eLife.63704."""
import pytest

from morie.fn.qrntcq import (efficacy_test_and_release,
                             gamma_generation_time, optimal_duration,
                             quarantine_efficacy, relative_utility,
                             utility)

G = gamma_generation_time()


def test_the_generation_time_density_is_normalised():
    tot = sum(0.5 * (G["density"][i] + G["density"][i + 1])
              * (G["t"][i + 1] - G["t"][i])
              for i in range(len(G["t"]) - 1))
    assert tot == pytest.approx(1.0, abs=1e-9)


def test_quarantine_over_the_whole_period_prevents_everything():
    assert quarantine_efficacy(0.0, 30.0, G)["efficacy"] == \
        pytest.approx(1.0, abs=1e-9)


def test_a_zero_length_quarantine_prevents_nothing():
    assert quarantine_efficacy(3.0, 3.0, G)["efficacy"] == \
        pytest.approx(0.0, abs=1e-12)


def test_efficacy_is_a_fraction():
    for tr in (4.0, 7.0, 12.0, 25.0):
        e = quarantine_efficacy(2.0, tr, G)["efficacy"]
        assert 0.0 <= e <= 1.0


def test_efficacy_increases_with_the_release_time():
    e = [quarantine_efficacy(3.0, t, G)["efficacy"]
         for t in (4.0, 6.0, 9.0, 13.0)]
    assert all(e[i] < e[i + 1] for i in range(len(e) - 1))


def test_efficacy_saturates():
    a = quarantine_efficacy(3.0, 14.0, G)["efficacy"]
    b = quarantine_efficacy(3.0, 20.0, G)["efficacy"]
    assert b - a < 0.03


def test_a_later_start_forfeits_more_transmission():
    a = quarantine_efficacy(1.0, 30.0, G)["pre_quarantine_mass"]
    b = quarantine_efficacy(6.0, 30.0, G)["pre_quarantine_mass"]
    assert b > a


def test_release_strategy_never_beats_full_detention():
    r = efficacy_test_and_release(3.0, 5.0, 7.0, 0.25,
                                  generation_time=G)
    assert r["efficacy"] <= r["bound"] + 1e-12


def test_a_perfect_test_matches_full_detention():
    r = efficacy_test_and_release(3.0, 5.0, 7.0, 0.0,
                                  generation_time=G)
    assert r["efficacy"] == pytest.approx(r["bound"], abs=1e-12)


def test_a_useless_test_matches_early_release():
    r = efficacy_test_and_release(3.0, 5.0, 7.0, 1.0,
                                  generation_time=G)
    assert r["efficacy"] == pytest.approx(r["efficacy_released"],
                                          abs=1e-12)


def test_testing_later_raises_efficacy():
    a = efficacy_test_and_release(3.0, 5.0, 7.0, 0.2,
                                  generation_time=G)["efficacy"]
    b = efficacy_test_and_release(3.0, 8.0, 10.0, 0.1,
                                  generation_time=G)["efficacy"]
    assert b > a


def test_utility_is_efficacy_over_days():
    assert utility(0.8, 4.0) == pytest.approx(0.2, abs=1e-15)


def test_relative_utility_ignores_the_infected_fraction():
    a = relative_utility(7.0, 10.0, t_Q=3.0, generation_time=G,
                         infected_fraction=0.01)["relative_utility"]
    b = relative_utility(7.0, 10.0, t_Q=3.0, generation_time=G,
                         infected_fraction=0.99)["relative_utility"]
    assert a == pytest.approx(b, abs=1e-15)


def test_the_utility_optimum_is_not_the_longest_duration():
    o = optimal_duration(t_Q=3.0, generation_time=G, t_max=20.0)
    assert o["optimal_t_R"] < 19.0


def test_a_release_before_the_start_is_refused():
    with pytest.raises(ValueError):
        quarantine_efficacy(6.0, 4.0, G)


def test_quarantine_before_exposure_is_refused():
    with pytest.raises(ValueError):
        quarantine_efficacy(-1.0, 5.0, G, t_E=0.0)


def test_an_out_of_range_false_negative_is_refused():
    with pytest.raises(ValueError):
        efficacy_test_and_release(3.0, 5.0, 7.0, -0.2,
                                  generation_time=G)


def test_a_test_before_quarantine_starts_is_refused():
    with pytest.raises(ValueError):
        efficacy_test_and_release(3.0, 2.0, 7.0, 0.2,
                                  generation_time=G)


def test_release_before_the_test_is_refused():
    with pytest.raises(ValueError):
        efficacy_test_and_release(3.0, 6.0, 5.0, 0.2,
                                  generation_time=G)


def test_a_non_positive_quarantine_length_is_refused():
    with pytest.raises(ValueError):
        utility(0.5, -1.0)


def test_a_non_positive_gamma_parameter_is_refused():
    with pytest.raises(ValueError):
        gamma_generation_time(scale=-1.0)
