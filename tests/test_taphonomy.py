# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the causal-taphonomy wrapper.

Guards the honesty invariants: the schema fabricates no rows, empty input is
refused, and the CATE path never emits ``sd/sqrt(n)`` as an SE -- ``"none"``
gives a point + dispersion only, ``"bootstrap"`` gives a valid SE + CI.
"""

import numpy as np
import pandas as pd
import pytest

from morie.taphonomy import taphonomy_preservation_delta, taphonomy_schema


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
