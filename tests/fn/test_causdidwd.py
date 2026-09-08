"""causdidwd -- ETWFE. Source: Wooldridge, J. M. (2025) Empirical
Economics 69, 2545-2587, doi:10.1007/s00181-025-02807-z."""
import pytest

from morie.fn import _array_core as np
from morie.fn.causdidwd import (aggregate, etwfe, imputation,
                                two_way_fixed_effects, two_way_mundlak)


def panel(noise=0.0, seed=5):
    rng = np.random.default_rng(seed)
    Y, U, T, FT, true = [], [], [], [], {}
    for i in range(60):
        g = {0: "3", 1: "5", 2: None}[i % 3]
        ai = float(rng.normal(0.0, 1.0))
        for t in range(1, 8):
            eff = 0.0
            if g is not None and t >= int(g):
                eff = 1.0 + 0.5 * (t - int(g))
                true[(g, str(t))] = eff
            Y.append(ai + 0.4 * t + eff
                     + (float(rng.normal(0.0, noise)) if noise else 0.0))
            U.append("u%d" % i)
            T.append(str(t))
            FT.append(g)
    return Y, U, T, FT, true


def treat_col(T, FT):
    return [[1.0 if (FT[i] is not None and int(T[i]) >= int(FT[i]))
             else 0.0] for i in range(len(T))]


def test_mundlak_reproduces_twfe_on_a_time_varying_regressor():
    Y, U, T, FT, _ = panel(noise=0.2)
    W = treat_col(T, FT)
    a = two_way_fixed_effects(Y, U, T, W)["coef"][0]
    b = two_way_mundlak(Y, U, T, W)["coef"][0]
    assert a == pytest.approx(b, abs=1e-7)


def test_mundlak_uses_far_fewer_columns():
    Y, U, T, FT, _ = panel()
    W = treat_col(T, FT)
    fe = two_way_fixed_effects(Y, U, T, W)
    tm = two_way_mundlak(Y, U, T, W)
    assert tm["n_columns"] < fe["n_columns"] / 5


def test_etwfe_is_exact_without_noise():
    Y, U, T, FT, true = panel(noise=0.0)
    e = etwfe(Y, U, T, FT)
    for cell, want in true.items():
        assert e["att"][cell] == pytest.approx(want, abs=1e-8)


def test_etwfe_has_one_coefficient_per_post_treatment_cell():
    Y, U, T, FT, true = panel()
    assert etwfe(Y, U, T, FT)["n_cells"] == len(true)


def test_imputation_matches_etwfe():
    Y, U, T, FT, _ = panel(noise=0.2)
    e = etwfe(Y, U, T, FT)
    i = imputation(Y, U, T, FT)
    for c in set(e["att"]) & set(i["att"]):
        assert e["att"][c] == pytest.approx(i["att"][c], abs=1e-6)


def test_imputation_is_exact_without_noise():
    Y, U, T, FT, true = panel(noise=0.0)
    i = imputation(Y, U, T, FT)
    for cell, want in true.items():
        assert i["att"][cell] == pytest.approx(want, abs=1e-8)


def test_imputation_fits_on_untreated_observations_only():
    Y, U, T, FT, _ = panel()
    i = imputation(Y, U, T, FT)
    assert i["n_untreated_used"] < len(Y)


def test_simple_aggregation_weights_sum_to_one():
    Y, U, T, FT, _ = panel()
    a = aggregate(etwfe(Y, U, T, FT), scheme="simple")
    assert sum(a["weights"].values()) == pytest.approx(1.0, abs=1e-12)


def test_event_aggregation_returns_a_profile():
    Y, U, T, FT, _ = panel()
    a = aggregate(etwfe(Y, U, T, FT), scheme="event")
    assert "profile" in a and len(a["profile"]) > 1


def test_a_single_period_panel_is_refused():
    Y, U, T, FT, _ = panel()
    with pytest.raises(ValueError):
        two_way_fixed_effects(Y[:8], U[:8], ["1"] * 8,
                              [[1.0]] * 8)


def test_an_adoption_period_outside_the_data_is_refused():
    Y, U, T, FT, _ = panel()
    with pytest.raises(ValueError):
        etwfe(Y, U, T, ["42"] * len(Y))


def test_a_panel_with_no_treated_units_is_refused():
    Y, U, T, FT, _ = panel()
    with pytest.raises(ValueError):
        etwfe(Y, U, T, [None] * len(Y))


def test_mismatched_lengths_are_refused():
    Y, U, T, FT, _ = panel()
    with pytest.raises(ValueError):
        two_way_mundlak(Y, U[:-1], T, treat_col(T, FT))


def test_an_unknown_aggregation_scheme_is_refused():
    Y, U, T, FT, _ = panel()
    with pytest.raises(ValueError):
        aggregate(etwfe(Y, U, T, FT), scheme="bayes")
