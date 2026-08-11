"""Anchored tests for smplqc.sample_qc (Marees 2018 / PLINK --het)."""

import math

from morie.fn.smplqc import sample_qc

NAN = float("nan")


def _hand_case():
    # G (3 x 4), one missing value.  Column allele frequencies:
    # col0: (0,2,0) p=1/3; col1: (1,2,1) p=2/3; col2: (2,1,0) p=1/2;
    # col3: (1,-,0) p=1/4.
    return [
        [0, 1, 2, 1],
        [2, 2, 1, NAN],
        [0, 1, 0, 0],
    ]


def test_hand_anchor_freq_callrate():
    res = sample_qc(_hand_case())
    f = res["freq"]
    assert abs(f[0] - 1 / 3) < 1e-12
    assert abs(f[1] - 2 / 3) < 1e-12
    assert abs(f[2] - 1 / 2) < 1e-12
    assert abs(f[3] - 1 / 4) < 1e-12
    assert res["callrate"] == [1.0, 0.75, 1.0]
    assert res["n_obs"] == [4, 3, 4]


def test_hand_anchor_F_het():
    """Hand arithmetic (PLINK F = (O - E)/(N - E)):

    per-column expected hom 1 - 2pq: col0 5/9, col1 5/9, col2 1/2,
    col3 5/8.  E(4 cols) = 5/9 + 5/9 + 1/2 + 5/8 = 2.2361111...,
    E(cols 0-2) = 1.6111111...
    ind0: O=2, F = (2 - 2.236111)/(4 - 2.236111) = -0.13385826...
    ind1: O=2, F = (2 - 1.611111)/(3 - 1.611111) = 0.28
    ind2: O=3, F = (3 - 2.236111)/(4 - 2.236111) = 0.43307086...
    het rates: 0.5, 1/3, 0.25.
    """
    res = sample_qc(_hand_case())
    e4 = 5 / 9 + 5 / 9 + 1 / 2 + 5 / 8
    e3 = 5 / 9 + 5 / 9 + 1 / 2
    assert abs(res["exp_hom"][0] - e4) < 1e-12
    assert abs(res["exp_hom"][1] - e3) < 1e-12
    assert res["obs_hom"] == [2, 2, 3]
    assert abs(res["F"][0] - (2 - e4) / (4 - e4)) < 1e-12
    assert abs(res["F"][1] - (2 - e3) / (3 - e3)) < 1e-12
    assert abs(res["F"][1] - 0.28) < 1e-12
    assert abs(res["F"][2] - (3 - e4) / (4 - e4)) < 1e-12
    assert abs(res["het_rate"][0] - 0.5) < 1e-12
    assert abs(res["het_rate"][1] - 1 / 3) < 1e-12
    assert abs(res["het_rate"][2] - 0.25) < 1e-12


def test_flags_and_pass():
    res = sample_qc(_hand_case())
    # ind1 call rate 0.75 < 0.98 -> flagged; het never flagged here
    # (max |dev| = 0.1389 < 3 * 0.12729)
    assert res["flag_callrate"] == [False, True, False]
    assert res["flag_het"] == [False, False, False]
    assert res["pass_qc"] == [True, False, True]
    assert res["estimate"] == 2.0
    hm = (0.5 + 1 / 3 + 0.25) / 3
    assert abs(res["het_mean"] - hm) < 1e-12
    hsd = math.sqrt(((0.5 - hm) ** 2 + (1 / 3 - hm) ** 2 + (0.25 - hm) ** 2) / 2)
    assert abs(res["het_sd"] - hsd) < 1e-12


def test_small_sample_multiplier():
    """Nei N/(N-1): col0 has N=3 observations -> c = 3/2, so the
    per-column expected hom becomes 1 - 2*(1/3)*(2/3)*(3/2) = 1/3."""
    res = sample_qc(_hand_case(), small_sample=True)
    # ind2 columns: c = (3/2, 3/2, 3/2, 2) for N = (3,3,3,2)
    e = (1 - 2 * (1 / 3) * (2 / 3) * 1.5) + (1 - 2 * (2 / 3) * (1 / 3) * 1.5) \
        + (1 - 2 * 0.5 * 0.5 * 1.5) + (1 - 2 * 0.25 * 0.75 * 2.0)
    assert abs(res["exp_hom"][2] - e) < 1e-12
