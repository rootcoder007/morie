"""spstp -- spatio-temporal point processes, Schabenberger & Gotway Sec. 9.5."""

import numpy as np
import pytest

from morie.fn.spstp import schabenberger_st_point_process

REGION = (0.0, 10.0, 0.0, 10.0)
SPAN = (0.0, 5.0)


def _poisson(n=600, seed=3):
    rs = np.random.RandomState(seed)
    return rs.uniform(0, 10, size=(n, 2)), rs.uniform(0, 5, n)


def test_intensity_is_n_over_space_time_volume():
    """eq (9.20) under FOST: lambda = N / (|A| |T|)."""
    pts, tt = _poisson(600)
    r = schabenberger_st_point_process(pts, REGION, SPAN, times=tt)
    assert r["volume"] == pytest.approx(500.0)
    assert r["intensity"] == pytest.approx(600 / 500.0)


def test_cstr_reference_moments():
    """N(A,T) ~ Poisson(lambda |A x T|); lambda_2 = lambda^2."""
    pts, tt = _poisson(600)
    ref = schabenberger_st_point_process(pts, REGION, SPAN, times=tt)["cstr"]
    assert ref["expected_count"] == pytest.approx(600.0)
    assert ref["variance"] == pytest.approx(ref["expected_count"])
    assert ref["second_order_intensity"] == pytest.approx(ref["intensity"] ** 2)


def test_dispersion_index_is_exactly_the_book_formula():
    """Sec. 3.3 eq (3.3), alternative form X^2 = (rc-1) s^2 / nbar.

    s^2 is the SAMPLE variance. Population variance differs by only
    (rc-1)/rc, which no p-value comparison would detect, so assert the
    arithmetic directly.
    """
    pts, tt = _poisson(600)
    r = schabenberger_st_point_process(pts, REGION, SPAN, times=tt,
                                       n_space_bins=3, n_time_bins=3)
    counts = np.asarray(r["cell_counts"], dtype=float).ravel()
    m = counts.size
    exact = (m - 1) * counts.var(ddof=1) / counts.mean()
    biased = (m - 1) * counts.var(ddof=0) / counts.mean()
    assert r["index_of_dispersion"] == pytest.approx(exact, rel=1e-12)
    assert r["index_of_dispersion"] != pytest.approx(biased, rel=1e-12)
    assert r["df"] == m - 1


def test_cstr_not_rejected_for_a_poisson_pattern():
    pts, tt = _poisson(600)
    r = schabenberger_st_point_process(pts, REGION, SPAN, times=tt)
    assert r["p_value"] > 0.05


def test_cstr_rejected_for_a_clustered_pattern():
    rs = np.random.RandomState(3)
    pts = np.vstack([rs.normal(2.0, 0.4, size=(300, 2)),
                     rs.normal(8.0, 0.4, size=(300, 2))]).clip(0, 10)
    tt = rs.uniform(0, 5, 600)
    r = schabenberger_st_point_process(pts, REGION, SPAN, times=tt)
    assert r["p_value"] < 1e-6


def test_marginals_integrate_back_to_n():
    """eqs (9.21) and (9.22)."""
    pts, tt = _poisson(600)
    r = schabenberger_st_point_process(pts, REGION, SPAN, times=tt,
                                       n_space_bins=4, n_time_bins=4)
    assert (r["marginal_spatial"].sum() * r["cell_area"]) == pytest.approx(600.0)
    assert (r["marginal_temporal"].sum()
            * r["time_bin_width"]) == pytest.approx(600.0)


def test_earthquake_process_carries_the_conditional_intensity_caveat():
    """Sec. 9.5.2: for an earthquake process lambda(s|t), lambda(t|s) are not meaningful."""
    pts, tt = _poisson(200)
    r = schabenberger_st_point_process(pts, REGION, SPAN, times=tt,
                                       process_type="earthquake")
    assert "conditional_note" in r


def test_birth_death_carries_the_identifiability_caveat():
    """Sec. 9.5.1: indistinguishable from a pattern sampled in time."""
    pts, tt = _poisson(200)
    r = schabenberger_st_point_process(pts, REGION, SPAN, times=tt,
                                       process_type="birth_death")
    assert "identifiability_note" in r


def test_low_cell_count_is_flagged_for_power():
    pts, tt = _poisson(200)
    r = schabenberger_st_point_process(pts, REGION, SPAN, times=tt,
                                       n_space_bins=2, n_time_bins=2)
    assert "power_note" in r


def test_missing_times_rejected():
    pts, _ = _poisson(50)
    with pytest.raises(ValueError, match="times"):
        schabenberger_st_point_process(pts, REGION, SPAN)


def test_unknown_process_type_rejected():
    pts, tt = _poisson(50)
    with pytest.raises(ValueError, match="process_type"):
        schabenberger_st_point_process(pts, REGION, SPAN, times=tt,
                                       process_type="wildfire")
