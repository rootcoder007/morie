# SPDX-License-Identifier: AGPL-3.0-or-later
"""Meta-analysis composites, at parity with rmorie.

The pooling anchors are closed-form: inverse-variance weights, Cochran's
Q, I-squared and the DerSimonian-Laird tau-squared are all computable by
hand from the inputs, so these assertions can fail.
"""

import math

import pytest

import morie

YS = [0.2, 0.4, 0.3]
SES = [0.1, 0.15, 0.12]


def test_fixed_effect_pooling_is_inverse_variance():
    p = morie.meta_pool(YS, SES)
    w = [1 / s ** 2 for s in SES]
    want = sum(wi * y for wi, y in zip(w, YS)) / sum(w)
    assert p["mean"] == pytest.approx(want, abs=1e-12)
    assert p["se"] == pytest.approx(math.sqrt(1 / sum(w)), abs=1e-12)
    assert p["z"] == pytest.approx(p["mean"] / p["se"], abs=1e-10)
    # the interval is symmetric about the mean at the given critical value
    lo, hi = p["ci"]
    assert lo == pytest.approx(p["mean"] - 1.96 * p["se"], abs=1e-12)
    assert hi == pytest.approx(p["mean"] + 1.96 * p["se"], abs=1e-12)


def test_cochran_q_and_i_squared_by_hand():
    p = morie.meta_pool(YS, SES)
    w = [1 / s ** 2 for s in SES]
    m = sum(wi * y for wi, y in zip(w, YS)) / sum(w)
    q = sum(wi * (y - m) ** 2 for wi, y in zip(w, YS))
    assert p["q"] == pytest.approx(q, abs=1e-10)
    assert p["df"] == len(YS) - 1
    assert p["i2"] == pytest.approx(max(0.0, (q - p["df"]) / q * 100),
                                    abs=1e-10)


def test_homogeneous_studies_get_no_random_effects_variance():
    # Q below its df means no more spread than sampling explains, so
    # DerSimonian-Laird truncates tau-squared at zero and the random
    # weights collapse onto the fixed ones.
    p = morie.meta_pool(YS, SES)
    assert p["tau2"] == 0.0
    assert p["i2"] == 0.0
    assert p["weights_random"] == pytest.approx(p["weights"])


def test_heterogeneous_studies_are_flagged_rather_than_averaged_away():
    # THE SUBSTANCE: reporting a pooled mean without the heterogeneity
    # beside it is how a meta-analysis overstates its case. These are
    # rmorie's values for the same input.
    h = morie.meta_pool([0.1, 0.9, 0.45, 1.4], [0.08, 0.10, 0.09, 0.12])
    assert h["mean"] == pytest.approx(0.5753135200, abs=1e-9)
    assert h["q"] == pytest.approx(95.0109994464, abs=1e-8)
    assert h["i2"] == pytest.approx(96.8424708534, abs=1e-8)
    assert h["tau2"] == pytest.approx(0.2806412488, abs=1e-9)
    # with real heterogeneity the random-effects weights must be smaller
    assert all(r < f for r, f in zip(h["weights_random"], h["weights"]))


def test_subgroups_split_q_into_within_and_between():
    g = morie.meta_pool([0.2, 0.4, 0.3, 0.5], [0.1, 0.15, 0.12, 0.2],
                        groups=["a", "a", "b", "b"])
    assert g["q_within"] == pytest.approx(1.9660633484, abs=1e-9)
    assert g["q_between"] == pytest.approx(0.4770891064, abs=1e-9)
    # the split must account for the total
    assert g["q_within"] + g["q_between"] == pytest.approx(g["q"], abs=1e-9)


def test_pooling_checks_its_inputs():
    with pytest.raises(ValueError, match="same length"):
        morie.meta_pool([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="strictly positive"):
        morie.meta_pool([0.1], [0.0])
    with pytest.raises(ValueError, match="must not be empty"):
        morie.meta_pool([], [])
    with pytest.raises(ValueError, match="groups"):
        morie.meta_pool(YS, SES, groups=["a"])


def test_effect_sizes_from_means_and_sds():
    # rmorie's values for the same input
    e = morie.meta_effect_sizes(m1=10, m2=8, s1=2, s2=2.5, n1=30, n2=32)
    assert e["s_pooled"] == pytest.approx(2.2721135535, abs=1e-9)
    assert e["d"] == pytest.approx(0.8802376963, abs=1e-9)
    assert e["g"] == pytest.approx(0.8691886875, abs=1e-9)
    assert e["se_g"] == pytest.approx(0.2658495559, abs=1e-9)
    # Hedges' g is the small-sample correction of d, so it shrinks
    assert abs(e["g"]) < abs(e["d"])


def test_effect_sizes_from_a_two_by_two_table():
    e = morie.meta_effect_sizes(a=20, b=80, c=10, d=90)
    assert e["rr"] == pytest.approx(2.0, abs=1e-10)
    assert e["or"] == pytest.approx(2.25, abs=1e-10)
    assert e["se_ln_rr"] == pytest.approx(0.3605551275, abs=1e-9)
    assert e["se_ln_or"] == pytest.approx(0.4166666667, abs=1e-9)


def test_absent_inputs_produce_absent_keys_not_zeros():
    # A fabricated zero is worse than a missing key: it looks like a
    # measured null.
    e = morie.meta_effect_sizes(n1=30, n2=32)
    assert "j" in e
    assert "d" not in e and "g" not in e and "rr" not in e
    assert morie.meta_effect_sizes() == {}


def test_conversion_constants_are_the_ones_the_literature_uses():
    cv = morie.meta_convert(ln_or=0.7, se_ln_or=0.2)
    assert cv["sd_logistic"] == pytest.approx(math.pi / math.sqrt(3.0),
                                              abs=1e-12)
    assert cv["sd_logistic"] == pytest.approx(1.8137993642, abs=1e-9)
    assert cv["d_logit"] == pytest.approx(0.3859302268, abs=1e-9)
    assert cv["d_cox"] == pytest.approx(0.4242424242, abs=1e-9)
    assert cv["se_d_logit"] == pytest.approx(0.1102657791, abs=1e-9)
    # the Cox route divides by 1.65 by definition
    assert cv["d_cox"] == pytest.approx(0.7 / 1.65, abs=1e-12)


def test_probit_and_correlation_routes():
    cv = morie.meta_convert(p1=0.6, p2=0.4, n1=100, n2=100)
    assert "d_probit" in cv and "se_d_probit" in cv
    assert cv["d_probit"] > 0                     # p1 > p2
    cr = morie.meta_convert(r=0.3, se_r=0.05)
    assert cr["fisher_z"] == pytest.approx(0.5 * math.log(1.3 / 0.7),
                                           abs=1e-12)
    assert "d_from_r" in cr and "se_d_from_se_r" in cr
