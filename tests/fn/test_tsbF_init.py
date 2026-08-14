"""tsbF initialisation routes. Sources: Teunter, Syntetos & Babai
(2011) EJOR 214(3), 606-615; Prak, Teunter, Babai, Boylan & Syntetos
(2021) Omega 104, 102481; Kourentzes (2014) IJPE 156, 180-190."""
import pytest

from morie.fn.tsbF import (croston_forecast, intermittent_forecast,
                           sba_forecast, tsb_forecast)

SERIES = [0.0, 5.0, 0.0, 0.0, 7.0, 0.0, 6.0, 0.0, 0.0, 0.0, 4.0, 0.0]


def test_global_init_starts_from_the_mean_of_positive_demands():
    r = tsb_forecast(SERIES, init="global")
    assert r["z_init"] == pytest.approx((5.0 + 7.0 + 6.0 + 4.0) / 4.0)


def test_heuristic_init_starts_from_the_first_positive_demand():
    r = tsb_forecast(SERIES, init="heuristic")
    assert r["z_init"] == pytest.approx(5.0)


def test_the_two_inits_give_different_paths():
    a = tsb_forecast(SERIES, init="global")["fitted"]
    b = tsb_forecast(SERIES, init="heuristic")["fitted"]
    assert any(abs(a[i] - b[i]) > 1e-9 for i in range(len(a)))


def test_known_init_uses_exactly_what_it_is_given():
    r = tsb_forecast(SERIES, init="known", z0=10.0, p0=0.25)
    assert r["z_init"] == pytest.approx(10.0)
    assert r["p_init"] == pytest.approx(0.25)


def test_known_init_without_z0_is_refused():
    with pytest.raises(ValueError):
        tsb_forecast(SERIES, init="known", p0=0.25)


def test_known_init_with_an_impossible_probability_is_refused():
    with pytest.raises(ValueError):
        tsb_forecast(SERIES, init="known", z0=1.0, p0=1.5)


def test_burn_in_shortens_the_reported_path_by_exactly_that_many():
    full = tsb_forecast(SERIES, burn_in=0)["fitted"]
    cut = tsb_forecast(SERIES, burn_in=4)
    assert len(cut["fitted"]) == len(full) - 4
    assert cut["fitted"][0] == pytest.approx(full[4])


def test_burn_in_keeps_the_full_path_available():
    r = tsb_forecast(SERIES, burn_in=4)
    assert len(r["fitted_full"]) == len(SERIES)


def test_burn_in_that_discards_everything_is_refused():
    with pytest.raises(ValueError):
        tsb_forecast(SERIES, burn_in=len(SERIES))


def test_negative_burn_in_is_refused():
    with pytest.raises(ValueError):
        tsb_forecast(SERIES, burn_in=-1)


def test_croston_accepts_a_known_interval():
    r = croston_forecast(SERIES, init="known", z0=6.0, x0=3.0)
    assert r["z_init"] == pytest.approx(6.0)
    assert r["x_init"] == pytest.approx(3.0)


def test_croston_refuses_an_interval_below_one():
    with pytest.raises(ValueError):
        croston_forecast(SERIES, init="known", z0=6.0, x0=0.5)


def test_sba_is_croston_times_the_deflator_under_any_init():
    for init in ("global", "heuristic"):
        c = croston_forecast(SERIES, alpha=0.2, init=init)["forecast"][0]
        s = sba_forecast(SERIES, alpha=0.2, init=init)["forecast"][0]
        assert s == pytest.approx(c * (1.0 - 0.1), abs=1e-13)


def test_dispatch_passes_the_init_through():
    a = intermittent_forecast(SERIES, method="tsb", init="heuristic")
    assert a["init"] == "heuristic"


def test_unknown_init_is_refused():
    with pytest.raises(ValueError):
        tsb_forecast(SERIES, init="mle")
