"""Rangayyan batch A: Chapter 3 statistics, delta/step, convolution.

Expected values are either closed forms of the distribution being
integrated (normal, uniform, exponential) or recomputed here straight
from the printed equation -- never copied from the implementation.
"""

import math

import pytest

from morie.fn.rng001 import pdfmean
from morie.fn.rng002 import pdfms
from morie.fn.rng003 import pdfvar
from morie.fn.rng004 import pdfskew
from morie.fn.rng005 import pdfkurt
from morie.fn.rng006 import diffent
from morie.fn.rng007 import smean
from morie.fn.rng009 import srms
from morie.fn.rng011 import shannon
from morie.fn.rng012 import noisemodel
from morie.fn.rng013 import meansum
from morie.fn.rng015 import ensmean
from morie.fn.rng018 import ensavg
from morie.fn.rng021 import covxy
from morie.fn.rng024 import diracdelta
from morie.fn.rng025 import deltaarea
from morie.fn.rng026 import deltalim
from morie.fn.rng027 import ustep
from morie.fn.rng028 import sifting
from morie.fn.rng029 import deltadecomp
from morie.fn.rng030 import contconv
from morie.fn.rng031 import contconvalt
from morie.fn.rng034 import kdelta
from morie.fn.rng035 import stepseq
from morie.fn.rng040 import rampfilt


def gauss(mu=0.0, sd=1.0):
    c = 1.0 / (sd * math.sqrt(2.0 * math.pi))
    return lambda v: c * math.exp(-0.5 * ((v - mu) / sd) ** 2)


def unif(a=0.0, b=1.0):
    return lambda v: 1.0 / (b - a) if a <= v <= b else 0.0


UGRID = [i / 4000.0 for i in range(4001)]


# ------------------------------------------------------------- eqs 3.1-3.6

def test_pdfmean_eq31_gaussian():
    # first-order moment of N(2, 1.5) is its mean
    r = pdfmean(gauss(2.0, 1.5), lower=2.0 - 12 * 1.5, upper=2.0 + 12 * 1.5)
    assert r["mean"] == pytest.approx(2.0, abs=1e-7)
    assert r["pdf_mass"] == pytest.approx(1.0, abs=1e-7)


def test_pdfmean_eq31_tabulated_uniform():
    r = pdfmean(unif(), x=UGRID)
    assert r["mean"] == pytest.approx(0.5, abs=1e-9)


def test_pdfms_eq32_is_second_moment_not_central():
    # E[eta^2] = sigma^2 + mu^2 = 1.5^2 + 2^2 = 6.25 for N(2, 1.5)
    r = pdfms(gauss(2.0, 1.5), lower=2.0 - 12 * 1.5, upper=2.0 + 12 * 1.5)
    assert r["ms"] == pytest.approx(6.25, abs=1e-6)
    assert r["variance_from_identity"] == pytest.approx(2.25, abs=1e-6)


def test_pdfvar_eq33_and_cv():
    # uniform(0,1): variance 1/12, sd 1/sqrt(12), cv = sd/mean
    r = pdfvar(unif(), x=UGRID)
    assert r["variance"] == pytest.approx(1.0 / 12.0, abs=1e-8)
    assert r["sd"] == pytest.approx(math.sqrt(1.0 / 12.0), abs=1e-8)
    assert r["cv"] == pytest.approx(math.sqrt(1.0 / 12.0) / 0.5, abs=1e-7)


def test_pdfvar_cv_is_none_at_zero_mean():
    # the book warns CV diverges as mu -> 0; symmetric density about 0
    g = [i / 1000.0 - 3.0 for i in range(6001)]
    r = pdfvar(gauss(0.0, 1.0), x=g)
    assert r["cv"] is None


def test_pdfskew_eq34_zero_for_symmetric():
    g = [i / 1000.0 - 8.0 for i in range(16001)]
    assert pdfskew(gauss(0.0, 1.0), x=g)["skewness"] == pytest.approx(
        0.0, abs=1e-8)


def test_pdfskew_eq34_exponential_is_two():
    # exponential(1) has skewness exactly 2
    g = [i / 500.0 for i in range(30001)]
    r = pdfskew(lambda v: math.exp(-v) if v >= 0 else 0.0, x=g)
    assert r["skewness"] == pytest.approx(2.0, abs=1e-4)


def test_pdfkurt_eq35_gaussian_is_three():
    g = [i / 1000.0 - 10.0 for i in range(20001)]
    r = pdfkurt(gauss(0.0, 1.0), x=g)
    assert r["kurtosis"] == pytest.approx(3.0, abs=1e-6)
    assert r["excess"] == pytest.approx(0.0, abs=1e-6)


def test_pdfkurt_eq35_uniform_is_nine_fifths():
    r = pdfkurt(unif(), x=UGRID)
    assert r["kurtosis"] == pytest.approx(1.8, abs=1e-6)
    assert r["excess"] == pytest.approx(1.8 - 3.0, abs=1e-6)


def test_diffent_eq36_gaussian_closed_form():
    # 0.5 log2(2 pi e sigma^2) bits
    sd = 2.0
    want = 0.5 * math.log2(2 * math.pi * math.e * sd * sd)
    g = [i / 500.0 - 24.0 for i in range(24001)]
    assert diffent(gauss(0.0, sd), x=g)["entropy"] == pytest.approx(
        want, abs=1e-6)


def test_diffent_eq36_can_be_negative():
    # uniform on (0, 0.5) has density 2 everywhere: H = log2(0.5) = -1 bit
    g = [i / 8000.0 for i in range(4001)]
    r = diffent(lambda v: 2.0 if 0.0 <= v <= 0.5 else 0.0, x=g)
    assert r["entropy"] == pytest.approx(-1.0, abs=1e-9)


# ----------------------------------------------------------- eqs 3.7-3.11

def test_smean_eq37():
    assert smean([1.0, 2.0, 6.0])["mean"] == pytest.approx(3.0)
    assert smean([1.0, 2.0, 6.0])["n"] == 3


def test_smean_rejects_empty():
    with pytest.raises(ValueError):
        smean([])


def test_srms_eqs38_310_divisor_is_N():
    x = [3.0, 4.0]
    r = srms(x)
    assert r["ms"] == pytest.approx(12.5)             # (9 + 16) / 2
    assert r["rms"] == pytest.approx(math.sqrt(12.5))
    assert r["sd"] == pytest.approx(0.5)              # N, not N-1
    assert r["ddof"] == 0


def test_shannon_eq311_uniform_is_log2L():
    r = shannon([0.25] * 4)
    assert r["entropy"] == pytest.approx(2.0)
    assert r["max_entropy"] == pytest.approx(2.0)


def test_shannon_eq311_degenerate_is_zero():
    assert shannon([1.0, 0.0, 0.0])["entropy"] == pytest.approx(0.0)


def test_shannon_eq311_quantizes_raw_values():
    # 8 values spread evenly over 4 equal-width levels -> 2 bits
    r = shannon([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1], levels=4)
    assert r["levels"] == 4
    assert r["entropy"] == pytest.approx(2.0)


def test_shannon_normalizes_unnormalized_counts():
    assert shannon([2, 2, 2, 2])["entropy"] == pytest.approx(2.0)


# ---------------------------------------------------------- eqs 3.12-3.22

def test_noisemodel_eq312_and_313():
    x = [1.0, 2.0, 3.0, 4.0]
    e = [0.5, -0.5, 0.5, -0.5]
    r = noisemodel(x, e)
    assert r["y"] == [1.5, 1.5, 3.5, 3.5]
    assert r["mean_observed"] == pytest.approx(r["mean_additive"])


def test_noisemodel_eq314_reports_the_gap_when_correlated():
    # x and eta perfectly correlated: eq 3.14 must NOT hold
    x = [1.0, 2.0, 3.0, 4.0]
    r = noisemodel(x, x)
    assert r["correlation"] == pytest.approx(1.0)
    assert r["variance_observed"] == pytest.approx(4.0 * r["variance_additive"]
                                                   / 2.0)


def test_noisemodel_eq314_holds_when_orthogonal():
    x = [1.0, -1.0, 1.0, -1.0]
    e = [1.0, 1.0, -1.0, -1.0]
    r = noisemodel(x, e)
    assert r["covariance"] == pytest.approx(0.0)
    assert r["variance_observed"] == pytest.approx(r["variance_additive"])


def test_noisemodel_rejects_length_mismatch():
    with pytest.raises(ValueError):
        noisemodel([1.0, 2.0], [1.0])


def test_meansum_eq313():
    r = meansum([1.0, 3.0], [10.0, 20.0], [0.5])
    assert r["mean"] == pytest.approx(2.0 + 15.0 + 0.5)
    assert r["component_means"] == [2.0, 15.0, 0.5]


def test_ensmean_eq315_over_records():
    recs = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
    r = ensmean(recs, index=1)
    assert r["mean"] == pytest.approx(20.0)
    assert r["m"] == 3
    # SE = sd / sqrt(M), sd computed with divisor M
    sd = math.sqrt(((10 - 20) ** 2 + 0 + (30 - 20) ** 2) / 3.0)
    assert r["se"] == pytest.approx(sd / math.sqrt(3.0))


def test_ensmean_rejects_out_of_range_index():
    with pytest.raises(IndexError):
        ensmean([[1.0], [2.0]], index=5)


def test_ensavg_eq318_is_pointwise_mean():
    recs = [[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]
    r = ensavg(recs)
    assert r["average"] == [1.0, 3.0, 5.0]
    assert r["m"] == 2 and r["n"] == 3


def test_ensavg_rejects_ragged_records():
    with pytest.raises(ValueError):
        ensavg([[1.0, 2.0], [1.0]])


def test_covxy_eqs321_322():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    r = covxy(x, y)
    # C_xy with divisor N: mean 2.5 / 5.0, deviations (-1.5..1.5)/(-3..3)
    want = (1.5 * 3.0 + 0.5 * 1.0 + 0.5 * 1.0 + 1.5 * 3.0) / 4.0
    assert r["covariance"] == pytest.approx(want)
    assert r["correlation"] == pytest.approx(1.0)


def test_covxy_ddof_one_matches_unbiased():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [4.0, 1.0, 3.0, 2.0]
    n = len(x)
    assert covxy(x, y, ddof=1)["covariance"] == pytest.approx(
        covxy(x, y)["covariance"] * n / (n - 1))


def test_covxy_correlation_none_for_constant():
    assert covxy([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["correlation"] is None


# ---------------------------------------------------------- eqs 3.24-3.35

def test_diracdelta_eq324_is_undefined_at_origin():
    r = diracdelta([-1.0, 0.0, 1.0])
    assert r["delta"] == [0.0, None, 0.0]
    assert r["undefined_at_zero"] is True


def test_diracdelta_rectangular_has_unit_area():
    w = 0.25
    r = diracdelta([-0.2, 0.0, 0.2, 1.0], width=w)
    assert r["height"] == pytest.approx(1.0 / w)
    assert r["delta"][3] == 0.0


def test_deltaarea_eq325_invariant_to_width():
    for w in (2.0, 0.5, 0.05):
        assert deltaarea(width=w)["area"] == pytest.approx(1.0, abs=1e-12)


def test_deltaarea_eq325_flags_a_non_delta():
    grid = [i / 100.0 - 1.0 for i in range(201)]
    vals = [0.5 if abs(v) <= 1.0 else 0.0 for v in grid]   # area 1.0
    assert deltaarea(t=grid, values=vals)["unit_area"] is True
    vals2 = [v * 2 for v in vals]                          # area 2.0
    assert deltaarea(t=grid, values=vals2)["unit_area"] is False


def test_deltalim_eq326_matches_the_printed_form():
    a = 0.4
    for t in (0.5, 1.0, 2.5):
        want = 0.5 * a * abs(t) ** (a - 1.0)
        assert deltalim([t], a)["values"][0] == pytest.approx(want)


def test_deltalim_eq326_diverges_at_origin():
    assert deltalim([0.0], 0.4)["values"][0] is None


def test_deltalim_area_tends_to_one():
    # integral over [-L, L] is L^a -> 1 as a -> 0
    areas = [deltalim([-3.0, 3.0], a)["area_symmetric"]
             for a in (0.8, 0.4, 0.2, 0.05)]
    assert areas == sorted(areas, reverse=True)
    assert areas[-1] == pytest.approx(3.0 ** 0.05)


def test_ustep_eq327_is_zero_at_origin():
    r = ustep([-1.0, 0.0, 1e-12, 1.0])
    assert r["u"] == [0.0, 0.0, 1.0, 1.0]


def test_ustep_shift():
    assert ustep([0.0, 1.0, 2.0], shift=1.0)["u"] == [0.0, 0.0, 1.0]


def test_sifting_eq328_selects_the_value():
    r = sifting(lambda t: t ** 2 + 1.0, 2.0, 0.0, 5.0)
    assert r["value"] == pytest.approx(5.0)
    assert r["inside"] is True


def test_sifting_eq328_is_zero_outside_and_on_the_limits():
    assert sifting(lambda t: 7.0, 9.0, 0.0, 5.0)["value"] == 0.0
    assert sifting(lambda t: 7.0, 5.0, 0.0, 5.0)["value"] == 0.0
    assert sifting(lambda t: 7.0, 0.0, 0.0, 5.0)["value"] == 0.0


def test_deltadecomp_eq329_weights_carry_the_integral():
    x = [1.0, 2.0, 3.0, 4.0]
    t = [0.0, 0.5, 1.0, 1.5]
    r = deltadecomp(x, t)
    # amplitudes recover exactly from weights / spacing
    assert r["reconstruction_error"] == pytest.approx(0.0, abs=1e-12)
    assert r["total_weight"] == pytest.approx(r["integral"])


def test_contconv_eq330_scales_by_dt():
    x = [1.0, 2.0]
    h = [1.0, 1.0]
    plain = [1.0, 3.0, 2.0]
    r = contconv(x, h, dt=0.5)
    assert r["y"] == pytest.approx([0.5 * v for v in plain])


def test_contconv_with_an_impulse_reproduces_the_response():
    h = [2.0, -1.0, 0.5]
    r = contconv([1.0], h, dt=1.0)
    assert r["y"] == pytest.approx(h)


def test_contconvalt_eq331_commutes_with_eq330():
    x = [1.0, -2.0, 3.0, 0.5]
    h = [0.25, 0.5, 0.25]
    a = contconv(x, h, dt=0.1)
    b = contconvalt(x, h, dt=0.1)
    assert b["y"] == pytest.approx(a["y"])
    assert b["commutes"] is True
    assert b["max_difference"] == pytest.approx(0.0, abs=1e-12)


def test_kdelta_eq334():
    r = kdelta(5)
    assert r["delta"] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert kdelta(5, shift=2, amplitude=1.5)["delta"] == [
        0.0, 0.0, 1.5, 0.0, 0.0]


def test_stepseq_eq335_is_one_at_origin():
    r = stepseq(list(range(-2, 3)))
    assert r["u"] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert r["value_at_origin"] == 1.0


def test_stepseq_first_difference_is_the_impulse():
    r = stepseq(list(range(-2, 4)), shift=1)
    # difference of u(n-1) is delta(n-1)
    assert r["first_difference"] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_continuous_and_discrete_steps_disagree_at_the_origin():
    # eq 3.27 uses t > 0, eq 3.35 uses n >= 0
    assert ustep([0.0])["u"][0] == 0.0
    assert stepseq([0])["u"][0] == 1.0


def test_rampfilt_eq342_taps_and_gain():
    r = rampfilt()
    assert r["n_taps"] == 501                      # 0.25 s at 2 kHz, plus t=0
    assert r["h"][0] == pytest.approx(2.5)         # 10 * 0.25
    assert r["h"][-1] == pytest.approx(0.0, abs=1e-12)
    # sum_{i=0}^{500} 10 (0.25 - i/2000) = 10 (125.25 - 62.625)
    assert r["gain"] == pytest.approx(626.25)
    assert sum(r["h_normalized"]) == pytest.approx(1.0)


def test_rampfilt_normalized_output_is_a_weighted_average():
    r = rampfilt([5.0] * 2000)
    # a constant input passes through a normalized weighted average intact
    assert r["y"][-1] == pytest.approx(5.0)


def test_rampfilt_rejects_bad_parameters():
    with pytest.raises(ValueError):
        rampfilt(fs=0.0)
    with pytest.raises(ValueError):
        rampfilt(duration=-1.0)
