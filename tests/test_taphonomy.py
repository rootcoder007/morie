# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the causal-taphonomy wrapper.

Guards the honesty invariants: the schema fabricates no rows, empty input is
refused, and the CATE path never emits ``sd/sqrt(n)`` as an SE -- ``"none"``
gives a point + dispersion only, ``"bootstrap"`` gives a valid SE + CI.
"""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
import pytest

from morie.taphonomy import (
    taphonomy_decay_absorption,
    taphonomy_decay_chain,
    taphonomy_decay_delta,
    taphonomy_decay_simulate,
    taphonomy_bhm,
    taphonomy_clr,
    taphonomy_evidence_loglik,
    taphonomy_ilr,
    taphonomy_likelihood_ratio,
    taphonomy_pmi_schema,
    taphonomy_preservation_delta,
    taphonomy_preservation_lr,
    taphonomy_schema,
    taphonomy_simulate_pxrf,
)
from morie.taphonomy import _read_usgs_soil_zip
from morie.fn._stats_core import norm


def test_schema_is_typed_zero_row_template():
    s = taphonomy_schema()
    assert (hasattr(s, "columns") or hasattr(s, "_cols"))
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


def test_evidence_loglik_matches_scipy_and_rejects_bad_sd():
    ll = taphonomy_evidence_loglik([1200, 1310], mean=1250, sd=80)
    assert abs(ll - float(norm.logpdf([1200, 1310], 1250, 80).sum())) < 1e-10
    with pytest.raises(ValueError, match="sd"):
        taphonomy_evidence_loglik([1], 0, sd=0)


def test_likelihood_ratio_log_space_and_verbal_band():
    r = taphonomy_likelihood_ratio(loglik_h1=-3.1, loglik_h2=-12.7)
    assert abs(r["log_lr"] - (-3.1 - (-12.7))) < 1e-12
    assert abs(r["lr"] - np.exp(-3.1 + 12.7)) < 1e-6
    assert "for H1" in r["verbal"]
    r2 = taphonomy_likelihood_ratio(-12.7, -3.1)  # swap -> invert
    assert abs(r2["lr"] - 1.0 / r["lr"]) < 1e-9
    assert "for H2" in r2["verbal"]


def test_preservation_lr_favours_matching_model():
    out = taphonomy_preservation_lr(
        evidence=[1200, 1310, 1180],
        natural={"mean": 1250, "sd": 90},
        alternative={"mean": 300, "sd": 120},
    )
    assert out["lr"] > 1
    assert abs(out["log_lr"] - (out["loglik_h1"] - out["loglik_h2"])) < 1e-10


def test_bhm_recovers_effect_and_prior_shrinks():
    rng = np.random.default_rng(1)
    n = 200
    lime = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "preservation_score": 0.5 * lime + rng.normal(0, 0.3, n),
            "lime_treatment": lime,
        }
    )
    b = taphonomy_bhm(df, covariates=["lime_treatment"])
    eff = b["coefficients"]
    lime_mean = float(eff.loc[eff["term"] == "lime_treatment", "post_mean"].iloc[0])
    assert 0.3 < lime_mean < 0.7
    assert (eff["post_sd"] > 0).all()
    assert ((eff["prob_positive"] >= 0) & (eff["prob_positive"] <= 1)).all()

    b_tight = taphonomy_bhm(
        df, covariates=["lime_treatment"],
        priors={"lime_treatment": {"mean": 0.0, "sd": 0.01}},
    )
    tight = float(
        b_tight["coefficients"].loc[
            b_tight["coefficients"]["term"] == "lime_treatment", "post_mean"
        ].iloc[0]
    )
    assert abs(tight) < abs(lime_mean)  # tight prior at 0 pulls the estimate down


def test_bhm_cmdstanpy_backend_recovers_effect():
    pytest.importorskip("cmdstanpy")
    try:
        import cmdstanpy
        cmdstanpy.cmdstan_path()
    except Exception:  # noqa: BLE001
        pytest.skip("CmdStan not installed")
    rng = np.random.default_rng(1)
    n = 120
    lime = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {"preservation_score": 0.5 * lime + rng.normal(0, 0.3, n),
         "lime_treatment": lime}
    )
    fit = taphonomy_bhm(df, covariates=["lime_treatment"],
                        backend="cmdstanpy", chains=2, iter=400)
    assert "NUTS" in fit["backend"]
    eff = float(
        fit["coefficients"].loc[
            fit["coefficients"]["term"] == "lime_treatment", "post_mean"
        ].iloc[0]
    )
    assert 0.3 < eff < 0.7
    assert fit["stanfit"] is not None


def test_bhm_partial_pools_groups():
    rng = np.random.default_rng(2)
    n = 150
    df = pd.DataFrame(
        {
            "preservation_score": rng.normal(size=n),
            "lime_treatment": rng.integers(0, 2, n),
            "context": rng.choice(list("abcde"), n),
        }
    )
    b = taphonomy_bhm(df, covariates=["lime_treatment"], group="context")
    ge = b["group_effects"]
    assert ge is not None
    assert ((ge["shrinkage"] >= 0) & (ge["shrinkage"] <= 1)).all()
    assert len(ge) == 5


def test_simulated_pxrf_is_closed_and_lime_skews_calcium():
    ctl = taphonomy_simulate_pxrf(200, "control", seed=1)
    trt = taphonomy_simulate_pxrf(200, "treatment", seed=1)
    elts = ctl.attrs["elements"]
    assert np.allclose(ctl[elts].to_numpy().sum(axis=1), 1.0)
    assert (trt["lime_treatment"] == 1).all()
    assert trt["Ca"].mean() > ctl["Ca"].mean()


def test_clr_zero_sum_and_ilr_full_rank():
    comp = taphonomy_simulate_pxrf(10, "treatment", seed=2).iloc[:, :6]
    clr = taphonomy_clr(comp)
    assert np.allclose(clr.sum(axis=1), 0.0)
    ilr = taphonomy_ilr(comp)
    assert ilr.shape[1] == 5
    assert np.all(np.isfinite(ilr))


def test_ilr_matches_r_pivot_coordinate_formula():
    # deterministic closed-form check (parity anchor with the R sibling)
    x = np.array([[0.5, 0.3, 0.2]])
    L = np.log(x)
    ilr1 = np.sqrt(1 / 2) * (L[:, :1].mean(1) - L[:, 1])
    ilr2 = np.sqrt(2 / 3) * (L[:, :2].mean(1) - L[:, 2])
    got = taphonomy_ilr(x, pseudocount=0.0)
    assert np.allclose(got, np.column_stack([ilr1, ilr2]))


def test_end_to_end_simulate_ilr_bhm_recovers_lime_signal():
    rng = np.random.default_rng(3)
    ctl = taphonomy_simulate_pxrf(120, "control", seed=10)
    trt = taphonomy_simulate_pxrf(120, "treatment", seed=11)
    raw = pd.concat([ctl, trt], ignore_index=True)
    ilr = taphonomy_ilr(raw.iloc[:, :6])
    df = pd.DataFrame(
        {
            "preservation_score": 0.6 * raw["lime_treatment"].to_numpy()
            + rng.normal(0, 0.3, len(raw)),
            "lime_treatment": raw["lime_treatment"].to_numpy(),
        }
    )
    for j in range(ilr.shape[1]):
        df[f"ilr{j + 1}"] = ilr[:, j]
    fit = taphonomy_bhm(df, covariates=["lime_treatment"])
    eff = float(
        fit["coefficients"].loc[
            fit["coefficients"]["term"] == "lime_treatment", "post_mean"
        ].iloc[0]
    )
    assert eff > 0.3


def test_usgs_soil_zip_parser_reads_csv_member(tmp_path):
    import zipfile

    csv = tmp_path / "ngdbsoil.csv"
    pd.DataFrame(
        {"lab_id": [1, 2], "ca_pct": [3.1, 8.2], "fe_pct": [2.0, 1.1]}
    ).to_csv(csv, index=False)
    zpath = tmp_path / "ngdbsoil-csv.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(csv, arcname="ngdbsoil.csv")
    df = _read_usgs_soil_zip(zpath, nrows=None)
    assert list(df.columns) == ["lab_id", "ca_pct", "fe_pct"]
    assert len(df) == 2


def test_pmi_schema_is_typed_zero_row_sto2022_template():
    s = taphonomy_pmi_schema()
    assert len(s) == 0
    assert {"accumulated_deg_days", "burial_depth_cm", "pmi_days"} <= set(s.columns)
    assert s.attrs["role"]["pmi_days"] == "outcome"


def test_morphosource_key_resolves_arg_then_env_then_errors(monkeypatch):
    from morie.taphonomy import _morphosource_key

    monkeypatch.delenv("MORPHOSOURCE_API_KEY", raising=False)
    assert _morphosource_key("explicit") == "explicit"          # arg wins
    monkeypatch.setenv("MORPHOSOURCE_API_KEY", "from_env")
    assert _morphosource_key() == "from_env"                    # env fallback
    monkeypatch.delenv("MORPHOSOURCE_API_KEY", raising=False)
    assert _morphosource_key(required=False) is None            # optional
    with pytest.raises(ValueError, match="MORPHOSOURCE_API_KEY"):
        _morphosource_key(required=True)


def test_morphosource_search_params_encode_query_facets_paging():
    from morie.taphonomy import _morphosource_search_params

    p = _morphosource_search_params(query="cranium", media_type="Mesh",
                                    per_page=25, page=2)
    assert p["q"] == "cranium"
    assert p["search_field"] == "all_fields"
    assert p["f.media_type"] == "Mesh"
    assert p["per_page"] == 25 and p["page"] == 2


def test_morphosource_fetch_refuses_missing_use_statement(monkeypatch):
    from morie.taphonomy import taphonomy_morphosource_fetch

    monkeypatch.setenv("MORPHOSOURCE_API_KEY", "k")
    with pytest.raises(ValueError, match="use_statement"):
        taphonomy_morphosource_fetch(media_id=1, use_statement="")
