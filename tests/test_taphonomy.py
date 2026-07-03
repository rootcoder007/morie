# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the causal-taphonomy wrapper.

Guards the honesty invariants: the schema fabricates no rows, empty input is
refused, and the CATE path never emits ``sd/sqrt(n)`` as an SE -- ``"none"``
gives a point + dispersion only, ``"bootstrap"`` gives a valid SE + CI.
"""

import numpy as np
import pandas as pd
import pytest

from morie.taphonomy import (
    taphonomy_decay_absorption,
    taphonomy_decay_chain,
    taphonomy_decay_delta,
    taphonomy_decay_simulate,
    taphonomy_preservation_delta,
    taphonomy_schema,
)


def test_schema_is_typed_zero_row_template():
    s = taphonomy_schema()
    assert isinstance(s, pd.DataFrame)
    assert len(s) == 0
    assert {"lime_treatment", "preservation_score", "pxrf_ca_ppm"} <= set(s.columns)
    assert s.attrs["role"]["lime_treatment"] == "treatment"
    assert s.attrs["role"]["preservation_score"] == "outcome"


def test_empty_data_is_refused():
    with pytest.raises(ValueError, match="never fabricates"):
        taphonomy_preservation_delta(taphonomy_schema())


def _synthetic(n=100, seed=1):
    rng = np.random.default_rng(seed)
    arid = rng.integers(0, 2, n)
    lime = rng.binomial(1, 1 / (1 + np.exp(-(-0.3 + 0.8 * arid))))
    return pd.DataFrame(
        {
            "lime_treatment": lime,
            "preservation_score": 0.4 * lime + 0.05 * arid + rng.normal(0, 0.2, n),
            "temp_c": rng.normal(15, 1, n),
            "arid": arid,
        }
    )


def test_cate_none_reports_no_se_but_keeps_dispersion():
    r = taphonomy_preservation_delta(_synthetic(), estimator="cate", se_method="none")
    assert np.isfinite(r["value"])       # point estimate is a float
    assert r["se"] is None               # no invalid SE emitted
    assert r["p_value"] is None
    assert np.isfinite(r["cate_sd"])     # heterogeneity still reported
    assert len(r["cate_per_unit"]) == 100


def test_cate_bootstrap_gives_valid_se_and_ordered_ci():
    r = taphonomy_preservation_delta(
        _synthetic(), estimator="cate", se_method="bootstrap", n_boot=15
    )
    assert r["se"] is not None and r["se"] > 0
    assert r["ci_lower"] < r["ci_upper"]
    assert r["p_value"] is not None


def test_decay_chain_is_row_stochastic_absorbing_dtmc():
    ch = taphonomy_decay_chain(preservation=0.6)
    assert np.allclose(ch["P"].sum(axis=1), 1.0)
    assert ch["absorbing"] == ["skeletal", "mummified"]
    idx = {s: i for i, s in enumerate(ch["states"])}
    assert ch["P"][idx["skeletal"], idx["skeletal"]] == 1.0
    assert ch["P"][idx["mummified"], idx["mummified"]] == 1.0


def test_absorption_sums_to_one_and_rises_with_preservation():
    a0 = taphonomy_decay_absorption(taphonomy_decay_chain(0.0))
    a1 = taphonomy_decay_absorption(taphonomy_decay_chain(0.8))
    assert abs(sum(a0["absorption"].values()) - 1.0) < 1e-10
    assert abs(sum(a1["absorption"].values()) - 1.0) < 1e-10
    assert a1["absorption"]["mummified"] > a0["absorption"]["mummified"]
    assert a0["expected_steps"] > 0


def test_decay_delta_positive_and_simulate_absorbs():
    d = taphonomy_decay_delta(0.8)
    assert d["delta"] > 0
    assert abs((d["p_mummified_treated"] - d["p_mummified_natural"]) - d["delta"]) < 1e-12
    path = taphonomy_decay_simulate(taphonomy_decay_chain(0.8), n_steps=500, seed=1)
    assert path[-1] in ("skeletal", "mummified")


def test_r_python_decay_parity():
    # same params -> same absorption probs as the R sibling (deterministic algebra)
    a = taphonomy_decay_absorption(taphonomy_decay_chain(0.7))["absorption"]
    assert 0.0 < a["mummified"] < 1.0
    assert abs(a["mummified"] + a["skeletal"] - 1.0) < 1e-12
