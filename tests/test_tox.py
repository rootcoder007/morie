# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.tox — forensic toxicology module (R parity: rmorie R/tox.R)."""

from __future__ import annotations

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
import pytest

from morie import tox


def test_matrix_schema_is_typed_zero_row():
    s = tox.tox_matrix_schema()
    assert (hasattr(s, "columns") or hasattr(s, "_cols"))
    assert len(s) == 0
    assert {"case_id", "analyte", "matrix", "conc", "lod", "loq"} <= set(s.columns)
    assert s.attrs["role"]["conc"] == "measurement"
    assert s.attrs["role"]["matrix"] == "matrix"


def test_calibration_recovers_known_curve_and_inverse_predicts():
    rng = np.random.default_rng(42)
    conc = np.array([0.05, 0.1, 0.5, 1, 5, 10])
    response = 1000 * conc + rng.normal(0, 1, conc.size)
    cal = tox.tox_calibration(conc, response, weights="none", response_unknown=250)
    assert cal["slope"] == pytest.approx(1000, rel=1e-3)
    assert cal["intercept"] == pytest.approx(0, abs=1)
    assert cal["r_squared"] > 0.999
    assert cal["conc_hat"] == pytest.approx(0.25, abs=1e-2)
    assert cal["flag"] == "quantifiable"


def test_calibration_flags_below_lod():
    rng = np.random.default_rng(1)
    conc = np.array([0.1, 0.5, 1, 5, 10, 50])
    response = 1000 * conc + rng.normal(0, 20, conc.size)
    cal = tox.tox_calibration(conc, response, weights="1/x^2")
    assert cal["lod"] > 0 and cal["loq"] > cal["lod"]
    below = tox.tox_calibration(
        conc, response, weights="1/x^2", response_unknown=1000 * cal["lod"] * 0.5
    )
    assert below["flag"] == "below_lod"


def test_calibration_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 3"):
        tox.tox_calibration([1, 2], [1, 2])
    with pytest.raises(ValueError, match="must be > 0"):
        tox.tox_calibration([0, 1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="must be"):
        tox.tox_calibration([1, 2, 3], [1, 2, 3], weights="bogus")


def test_pmr_ratio_classifies_redistribution():
    assert tox.tox_pmr_ratio(1.0, 1.0)["redistribution"] == "minimal"
    assert tox.tox_pmr_ratio(1.8, 1.0)["redistribution"] == "modest"
    assert tox.tox_pmr_ratio(4.0, 1.0)["redistribution"] == "significant"
    assert tox.tox_pmr_ratio(2.4, 0.8)["cp_ratio"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="> 0"):
        tox.tox_pmr_ratio(1, 0)


def test_antemortem_lr_direction():
    res = tox.tox_antemortem_lr(
        2.0, {"mean": 2.0, "sd": 0.5}, {"mean": 0.1, "sd": 0.3}
    )
    assert res["lr"] > 1
    assert "antemortem" in res["interpretation"]
    res2 = tox.tox_antemortem_lr(
        0.1, {"mean": 2.0, "sd": 0.5}, {"mean": 0.1, "sd": 0.3}
    )
    assert res2["lr"] < 1


def test_matrix_reliability_ranks_protected_above_blood():
    r = tox.tox_matrix_reliability(submersion_days=14, decomp_stage=3)
    assert list(r["rank"]) == list(range(1, len(r) + 1))
    v = r.index[r["matrix"] == "vitreous_humour"][0]
    c = r.index[r["matrix"] == "central_blood"][0]
    assert v < c
    dry = tox.tox_matrix_reliability(matrix=["peripheral_blood"])
    wet = tox.tox_matrix_reliability(matrix=["peripheral_blood"], submersion_days=30)
    assert wet["reliability"][0] < dry["reliability"][0]
    with pytest.raises(ValueError, match="unknown matrix"):
        tox.tox_matrix_reliability(matrix=["plasma"])


def test_left_censor_impute():
    out = tox.tox_left_censor_impute([0.4, np.nan, 0.9, 0.02], lod=0.05)
    assert np.allclose(out["imputed"], [0.4, 0.025, 0.9, 0.025])
    assert list(out["censored"]) == [False, True, False, True]
    assert out["fraction_censored"] == 0.5
    assert tox.tox_left_censor_impute([np.nan], lod=0.1, method="sqrt2")[
        "imputed"
    ][0] == pytest.approx(0.1 / np.sqrt(2))
    with pytest.raises(ValueError, match="> 0"):
        tox.tox_left_censor_impute([1], lod=0)


def test_ethanol_congeners_adjudication():
    assert tox.tox_ethanol_congeners(1.2, n_propanol=0.08)["verdict"] == "postmortem_production"
    assert tox.tox_ethanol_congeners(1.2, etg=3.5)["verdict"] == "antemortem"
    assert tox.tox_ethanol_congeners(1.2)["verdict"] == "indeterminate"
    # EtG precedence even with congeners present
    assert tox.tox_ethanol_congeners(1.2, n_propanol=0.1, etg=2)["verdict"] == "antemortem"
    with pytest.raises(ValueError, match=">= 0"):
        tox.tox_ethanol_congeners(-1)
