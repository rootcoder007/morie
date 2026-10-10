"""Module-level execution surface for MORIE dataset analyses."""

from __future__ import annotations

import logging
import math as _math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd
from morie.fn._glm_core import NormalIndPower

from .causal import run_ebac_selection_ipw_analysis, run_propensity_ipw_analysis
from .cpads import canonicalize_cpads_frame, has_raw_cpads_columns
from .data import DatasetRegistry
from .investigation import (
    compare_nested_logistic_models,
    run_treatment_effects_analysis,
    run_weighted_logistic_analysis,
)


def proportion_effectsize(prop1, prop2):
    """Cohen's h = 2 asin(sqrt(p1)) - 2 asin(sqrt(p2)) (statsmodels-free)."""
    return 2.0 * _math.asin(_math.sqrt(prop1)) - 2.0 * _math.asin(_math.sqrt(prop2))


def _find_cpads_csv() -> str:
    """Locate a CPADS CSV.

    Resolution order:
      1. Real CPADS PUMF microdata at the documented project-root
         path (data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv).
         The Canadian Postsecondary Alcohol and Drug Survey 2021-2022 PUMF
         is OPEN DATA under the Open Government Licence - Canada; no
         subscription is required. Download it from
         <https://open.canada.ca/data/en/dataset/736fa9b2-62e4-4e31-aea4-51869605b363>.
         It is not bundled because the CSV is ~39 MB (40,931 data rows),
         far past what a CRAN package may ship, so rmoriedata carries the
         provenance record instead (SHA-256 and the direct URL, in
         inst/extdata/cpads_data_provenance.json) for fetch-and-verify.
      2. A 1,200-row synthetic CPADS-shaped frame shipped inside the
         wheel at morie/data/cpads_synthetic.csv.  This lets fresh
         users run `morie run-module power-design` on their first
         install without a manual download step, with a loud
         "synthetic data" warning so results aren't mistaken for
         analyses of the real survey.

    The function always returns a path; existence is verified at
    load time by load_cpads_analysis_data().
    """
    from .data import _project_root

    root = _project_root()
    real = root / "data" / "datasets" / "oc" / "CPADS" / "2021-2022" / "cpads-2021-2022-pumf2.csv"
    if real.exists():
        return str(real)

    # Fallback: shipped synthetic CSV (always exists; bundled in package-data)
    synthetic = Path(__file__).resolve().parent / "data" / "cpads_synthetic.csv"
    if synthetic.exists():
        return str(synthetic)

    # Neither found — return the real path so the eventual error
    # tells the user where the real CSV is expected.
    return str(real)


DEFAULT_CPADS_CSV = _find_cpads_csv()


def _is_synthetic_cpads_path(p: str | Path) -> bool:
    """True if the resolved path points at the shipped synthetic CSV."""
    return Path(p).name == "cpads_synthetic.csv"


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    description: str
    output_files: tuple[str, ...]


MODULE_SPECS = {
    "data-wrangling": ModuleSpec(
        name="data-wrangling",
        description="Canonicalize and validate the real CPADS PUMF input.",
        output_files=("data_na_summary.csv", "data_wrangling_log.csv"),
    ),
    "descriptive-statistics": ModuleSpec(
        name="descriptive-statistics",
        description="Survey-weighted prevalence and probability summaries.",
        output_files=("binomial_summaries.csv", "binomial_summaries_survey_weighted.csv", "probability_estimates.csv"),
    ),
    "distribution-tests": ModuleSpec(
        name="distribution-tests",
        description="Distributional diagnostics, correlations, and CLT checks.",
        output_files=("distribution_tests.csv", "alcohol_correlation_matrix.csv", "clt_convergence.csv"),
    ),
    "frequentist-inference": ModuleSpec(
        name="frequentist-inference",
        description="Frequentist prevalence, effect-size, and hypothesis-test outputs.",
        output_files=(
            "frequentist_heavy_drinking_prevalence_ci.csv",
            "frequentist_effect_sizes.csv",
            "frequentist_hypothesis_tests.csv",
        ),
    ),
    "bayesian-inference": ModuleSpec(
        name="bayesian-inference",
        description="Beta-binomial Bayesian summaries for key CPADS endpoints.",
        output_files=(
            "bayesian_posterior_summaries.csv",
            "bayesian_bayes_factors.csv",
            "bayesian_vs_frequentist_ci.csv",
        ),
    ),
    "power-design": ModuleSpec(
        name="power-design",
        description="Survey-weighted power planning summaries from real CPADS data.",
        output_files=(
            "power_summary.csv",
            "power_two_proportion_gender.csv",
            "power_one_proportion_grid.csv",
            "power_ebac_endpoint_anchors.csv",
            "power_gpower_reference_two_group.csv",
            "power_interaction_assumptions.csv",
            "power_interaction_feasibility_flags.csv",
            "power_interaction_group_allocations.csv",
            "power_interaction_imbalance_penalty.csv",
            "power_interaction_pairwise_details.csv",
            "power_interaction_sample_size_targets.csv",
            "randomization_block_blueprints.csv",
            "randomization_schedule_example_heavy_drinking_30d.csv",
            "randomization_schedule_example_ebac_legal.csv",
            "randomization_schedule_example_ebac_tot.csv",
        ),
    ),
    "logistic-models": ModuleSpec(
        name="logistic-models",
        description="Survey-weighted logistic models for heavy drinking.",
        output_files=(
            "logistic_odds_ratios.csv",
            "logistic_interaction_odds_ratios.csv",
            "logistic_interaction_tests.csv",
            "logistic_smote_status.csv",
            "logistic_smote_odds_ratios.csv",
        ),
    ),
    "model-comparison": ModuleSpec(
        name="model-comparison",
        description="Nested model comparison for heavy drinking models.",
        output_files=(
            "model_comparison_summary.csv",
            "model_comparison_full_coefs.csv",
            "model_comparison_interaction.csv",
            "model_comparison_wald_tests.csv",
        ),
    ),
    "regression-models": ModuleSpec(
        name="regression-models",
        description="Weighted regression models for eBAC outcomes.",
        output_files=("regression_coefficients.csv", "regression_model_comparison.csv"),
    ),
    "propensity-scores": ModuleSpec(
        name="propensity-scores",
        description="Propensity/IPW workflow for cannabis and heavy drinking.",
        output_files=("ipw_results.csv", "ipw_diagnostics.csv"),
    ),
    "causal-estimators": ModuleSpec(
        name="causal-estimators",
        description="Causal-estimator comparison across IPW, outcome-regression, and AIPW.",
        output_files=("causal_estimator_comparison.csv",),
    ),
    "treatment-effects": ModuleSpec(
        name="treatment-effects",
        description="ATE/ATT/ATC and subgroup treatment-effect summaries.",
        output_files=("treatment_effects_summary.csv", "cate_subgroup_estimates.csv"),
    ),
    "dag-specification": ModuleSpec(
        name="dag-specification",
        description="DAG and official-document alignment checklist outputs.",
        output_files=("official_doc_alignment_checklist.csv",),
    ),
    "meta-synthesis": ModuleSpec(
        name="meta-synthesis",
        description="Narrative synthesis outputs for study integration and interpretation.",
        output_files=("10_methods_results_paper.md", "11_interpretation.md"),
    ),
    "ebac-core": ModuleSpec(
        name="ebac-core",
        description="Core eBAC weighted, missingness, and model outputs.",
        output_files=(
            "ebac_data_quality_checks.csv",
            "ebac_distribution_unweighted.csv",
            "ebac_model_samples.csv",
            "ebac_weighted_summaries.csv",
            "ebac_missingness_weighted.csv",
            "ebac_missingness_or.csv",
            "ebac_missingness_or_eligible_drinkers.csv",
            "ebac_logistic_or_primary.csv",
            "ebac_linear_coefficients_primary.csv",
            "ebac_logistic_or_sensitivity_with_heavy.csv",
            "ebac_linear_coefficients_sensitivity_with_heavy.csv",
        ),
    ),
    "ebac-selection-adjustment-ipw": ModuleSpec(
        name="ebac-selection-adjustment-ipw",
        description="Selection-adjusted eBAC IPW workflow.",
        output_files=(
            "ebac_ipw_weight_diagnostics.csv",
            "ebac_ipw_logistic_or.csv",
            "ebac_ipw_linear_coefficients.csv",
            "ebac_ipw_cannabis_comparison.csv",
            "ebac_ipw_observation_model_or.csv",
            "ebac_ipw_covariate_balance.csv",
            "ebac_final_ipw_diagnostics.csv",
            "ebac_final_ipw_or.csv",
            "ebac_final_ipw_linear.csv",
            "ebac_final_ipw_comparison.csv",
        ),
    ),
    "ebac-integrations": ModuleSpec(
        name="ebac-integrations",
        description="Integrated eBAC final-summary outputs.",
        output_files=(
            "ebac_final_domain_samples.csv",
            "ebac_final_formula_input_audit.csv",
            "ebac_final_formula_validation.csv",
            "ebac_final_interaction_tests.csv",
            "ebac_final_weighted_descriptives.csv",
            "ebac_final_weighted_linear.csv",
            "ebac_final_weighted_or.csv",
            "ebac_final_smote_compare.csv",
            "ebac_final_smote_or.csv",
            "ebac_final_smote_status.csv",
            "ebac_final_causal_effects.csv",
            "ebac_final_cate.csv",
            "ebac_final_consistency_checks.csv",
            "ebac_final_crosswalk_previous.csv",
            "ebac_final_dml_results.csv",
            "ebac_final_dml_status.csv",
            "ebac_final_key_summary.csv",
            "ebac_final_user_guide_variable_map.csv",
            "ebac_final_variable_audit.csv",
        ),
    ),
    "ebac-gender-smote-sensitivity": ModuleSpec(
        name="ebac-gender-smote-sensitivity",
        description="eBAC interaction and SMOTE-sensitivity status outputs.",
        output_files=(
            "ebac_gender_interaction_svy_or.csv",
            "ebac_gender_interaction_tests.csv",
            "ebac_gender_marginal_probs.csv",
            "ebac_smote_status.csv",
            "ebac_smote_or.csv",
            "ebac_smote_compare.csv",
        ),
    ),
    "figures": ModuleSpec(
        name="figures",
        description="Figure exports for the documented analysis workflow.",
        output_files=(
            "figures/balance_plot.pdf",
            "figures/bayesian_prior_posterior.pdf",
            "figures/bayesian_prior_posterior.png",
            "figures/bayesian_vs_frequentist_ci.pdf",
            "figures/bayesian_vs_frequentist_ci.png",
            "figures/binge_by_demographics.pdf",
            "figures/binge_by_demographics.png",
            "figures/binge_by_mental_health.pdf",
            "figures/binge_by_mental_health.png",
            "figures/cate_forest_plot.pdf",
            "figures/cate_forest_plot.png",
            "figures/dag_heavy_drinking.pdf",
            "figures/qq_plots.pdf",
        ),
    ),
    "tables": ModuleSpec(
        name="tables",
        description="HTML table exports for the documented analysis workflow.",
        output_files=("table1.html",),
    ),
    "final-report": ModuleSpec(
        name="final-report",
        description="Final report and output-audit summaries.",
        output_files=(
            "ebac_final_output_coverage.csv",
            "ebac_final_output_shapes.csv",
            "ebac_final_script_run_status.csv",
            "ebac_final_audit_checks.csv",
            "ebac_final_user_guide_excerpt.txt",
        ),
    ),
    "otis-analysis": ModuleSpec(
        name="otis-analysis",
        description="OTIS restrictive-confinement analysis (descriptives, alert combos, cross-fitted DML, trends) on the bundled b01 sample.",
        output_files=("otis_descriptives.csv", "otis_alert_combos.csv", "otis_dml_results.csv", "otis_trends.csv"),
    ),
    "mapq-psychometrics": ModuleSpec(
        name="mapq-psychometrics",
        description="MAPQ psychometric validation (reliability, factor loadings) + DML (gender -> KS) on a synthetic MAPQII panel.",
        output_files=("mapq_reliability.csv", "mapq_factor_loadings.csv", "mapq_dml_results.csv"),
    ),
}


def list_modules() -> list[dict[str, object]]:
    """Return the currently implemented CPADS module surface."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "output_files": list(spec.output_files),
        }
        for spec in MODULE_SPECS.values()
    ]


def load_cpads_analysis_data(
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    *,
    column_mapping: dict[str, str] | None = None,
    auto_map: bool = False,
) -> pd.DataFrame:
    """Load and canonicalize a CPADS CSV into MORIE analysis columns.

    When the caller does not pass an explicit `cpads_csv` and the real
    Statistics Canada PUMF file is not on disk, this falls back to the
    bundled 1,200-row synthetic frame and emits a warning.  Outputs
    produced from the synthetic frame are useful for testing and demo
    purposes only; do not interpret them as findings about the real
    Canadian population.

    Schema-agnostic mode
    --------------------
    Pass ``column_mapping={"my_wt": "weight", "drinks_yn":
    "alcohol_past12m", ...}`` to rename your dataset's columns into
    morie's canonical names BEFORE the CPADS contract validator runs.
    Pass ``auto_map=True`` to have morie infer the mapping with a
    fuzzy match against a synonym table (`morie.schema.infer_mapping`);
    a warning will surface the inferred mapping so you can review
    before trusting it.
    """
    import warnings

    resolved = Path(cpads_csv).expanduser().resolve()
    if _is_synthetic_cpads_path(resolved) and not (column_mapping or auto_map):
        from .data import cached_cpads

        cached = cached_cpads()
        if cached is not None:
            # the real PUMF pulled earlier (`morie pull cpads`) beats the toy frame
            return canonicalize_cpads_frame(cached) if has_raw_cpads_columns(cached) else cached
    if _is_synthetic_cpads_path(resolved):
        warnings.warn(
            "morie: using the SHIPPED SYNTHETIC CPADS frame "
            f"({resolved}). This is a 1,200-row toy dataset with the "
            "correct schema but random data, intended for first-run "
            "demos. Get the real Statistics Canada CPADS PUMF once with "
            "`morie pull cpads` (cached, used by default afterwards), or "
            "run modules with --dataset ocp21 / cpads_csv=... .",
            UserWarning,
            stacklevel=2,
        )
    # ── Schema-agnostic remap path ────────────────────────────────
    # When the caller supplies (or asks us to infer) a column mapping,
    # we bypass DatasetRegistry.load()'s strict-validation gate — it
    # would reject the user's pre-rename column names before we could
    # remap them.  We still run `canonicalize_cpads_frame` afterwards,
    # which re-validates against the canonical contract.
    if column_mapping or auto_map:
        raw = pd.read_csv(resolved)
        if auto_map and column_mapping is None:
            from .cpads import CPADS_REQUIRED_VARIABLES
            from .schema import infer_mapping

            column_mapping, scores = infer_mapping(raw, canonical=CPADS_REQUIRED_VARIABLES)
            warnings.warn(
                "morie: auto_map=True inferred column mapping "
                f"{column_mapping} (scores {scores}). Pass column_mapping=... "
                "explicitly to override or to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
        if column_mapping:
            from .schema import apply_mapping

            raw = apply_mapping(raw, column_mapping)
        return canonicalize_cpads_frame(raw)

    # ── Standard path (data already in canonical schema) ──────────
    registry = DatasetRegistry(data_dir=".")
    registry.register_local_cpads(resolved, name="cpads_local")
    raw = registry.load("cpads_local")
    return canonicalize_cpads_frame(raw)


# The R-bridge shim ships inside the package (morie/rscripts/run_modules.R) so
# it resolves from __file__ and works the same across every install layout
# (source checkout, wheel, or bundled app).
_R_MODULE_SHIM = Path(__file__).resolve().parent / "rscripts" / "run_modules.R"


def _rscript_bin() -> str | None:
    return shutil.which("Rscript")


def _load_written_outputs(module_name: str, output_dir: Path) -> dict[str, object]:
    outputs: dict[str, object] = {}
    for filename in MODULE_SPECS[module_name].output_files:
        path = output_dir / filename
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            outputs[path.stem] = pd.read_csv(path)
        elif path.suffix.lower() == ".txt":
            outputs[path.stem] = path.read_text()
    return outputs


def _run_r_module(
    module_name: str,
    *,
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    rscript = _rscript_bin()
    if rscript is None:
        raise RuntimeError("Rscript is not available on PATH.")
    _tmp_ctx = None
    if output_dir is None:
        _tmp_ctx = tempfile.TemporaryDirectory(prefix=f"morie-{module_name}-")
        output_dir = Path(_tmp_ctx.name)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Rscript runs with cwd=output_dir, so any relative path we forward would
    # resolve against the OUTPUT dir, not the caller's cwd. Make cpads_csv
    # absolute here so it works regardless of where output lands (N1).
    cpads_csv = Path(cpads_csv).expanduser().resolve()
    try:
        if not _R_MODULE_SHIM.exists():
            raise RuntimeError(
                f"R-bridge shim not found at {_R_MODULE_SHIM}. The package may be "
                "installed incompletely (rscripts/run_modules.R must ship in the wheel)."
            )
        cmd = [
            rscript,
            str(_R_MODULE_SHIM),
            f"--modules={module_name}",
            f"--cpads-csv={cpads_csv}",
            f"--output-dir={output_dir}",
        ]
        env = dict(os.environ, MORIE_R_PACKAGE=_R_PACKAGE) if _R_PACKAGE else None
        proc = _sp().run(cmd, cwd=str(output_dir), check=False, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"R-backed module run failed for {module_name}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        import warnings

        # the R module says when it ran on stand-in data ("... synthetic ... not findings"): pass that on
        for line in (proc.stderr or "").splitlines():
            if "synthetic" in line.lower():
                warnings.warn(line.strip(), UserWarning, stacklevel=2)
        return _load_written_outputs(module_name, output_dir)
    finally:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()


def _write_outputs(outputs: dict[str, pd.DataFrame], output_dir: str | Path | None) -> dict[str, pd.DataFrame]:
    if output_dir is None:
        return outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(output_dir / f"{name}.csv", index=False)
    return outputs


_GENDER_LABELS = {1: "Female", 2: "Male", 3: "Non-binary"}


def _gender_label(g) -> str:
    """The CPADS code as its label (1 -> Female); a frame that already holds labels keeps them."""
    try:
        return _GENDER_LABELS.get(int(g), str(g))
    except (TypeError, ValueError):
        return str(g)


def _two_proportion_rows(df, outcome: str, g1, g2) -> list[dict]:
    """One power row for two gender groups, as the R route writes it.

    p1, p2 are weighted prevalences; h is Cohen's h; n_eq the per-group n for 80% power
    at alpha 0.05 with equal allocation, 2((z_a + z_b)/h)^2; power_srs the power the observed
    n1, n2 give, Phi(|h| / sqrt(1/n1 + 1/n2) - z_a); the design-effect columns inflate
    both by Kish's deff = n sum(w^2) / (sum w)^2 over the two groups' weights.
    """
    from statistics import NormalDist

    nd = NormalDist()
    rows: list[dict] = []
    groups = list(df["gender"])
    ys = list(df[outcome])
    ws = list(df["weight"])

    def _pick(g):
        out = []
        for gv, yv, wv in zip(groups, ys, ws):
            if gv == g and yv == yv and yv is not None and wv == wv and wv is not None:
                out.append((float(yv), float(wv)))
        return out

    a, b = _pick(g1), _pick(g2)
    if not a or not b:
        return rows
    p1 = sum(y * w for y, w in a) / sum(w for _, w in a)
    p2 = sum(y * w for y, w in b) / sum(w for _, w in b)
    h = 2 * _math.asin(_math.sqrt(p1)) - 2 * _math.asin(_math.sqrt(p2))
    z_a, z_b = nd.inv_cdf(0.975), nd.inv_cdf(0.80)
    w_all = [w for _, w in a] + [w for _, w in b]
    deff = len(w_all) * sum(w * w for w in w_all) / sum(w_all) ** 2
    n1, n2 = len(a), len(b)
    n_eq = 2 * ((z_a + z_b) / abs(h)) ** 2 if h != 0 else float("nan")
    se_unit = _math.sqrt(1 / n1 + 1 / n2)
    rows.append(
        {
            "group1": _gender_label(g1),
            "group2": _gender_label(g2),
            "p1": p1,
            "p2": p2,
            "h": h,
            "n1": n1,
            "n2": n2,
            "n_eq": n_eq,
            "power_srs": nd.cdf(abs(h) / se_unit - z_a),
            "n_eq_eff": n_eq * deff,
            "power_deff": nd.cdf(abs(h) / (se_unit * _math.sqrt(deff)) - z_a),
            "analysis_mode": "observational",
            "power_scope": outcome,
        }
    )
    return rows


def run_power_design_module(
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Build real-data power summaries from the CPADS CSV."""
    frame = load_cpads_analysis_data(cpads_csv)
    analysis = frame.dropna(subset=["heavy_drinking_30d", "gender", "weight"]).copy()
    gender_summary = (
        analysis.groupby("gender", dropna=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "weighted_prevalence": float((g["heavy_drinking_30d"] * g["weight"]).sum() / g["weight"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    pair_rows: list[dict[str, float]] = []
    power_grid_rows: list[dict[str, float]] = []
    power_tool = NormalIndPower()
    sample_sizes = [200, 400, 600, 800, 1000, 1500, 2000]

    if len(gender_summary) >= 2:
        ref = gender_summary.iloc[0]
        for _, other in gender_summary.iloc[1:].iterrows():
            effect = float(proportion_effectsize(ref["weighted_prevalence"], other["weighted_prevalence"]))
            for n_total in sample_sizes:
                n_per_group = n_total / 2
                power = float(power_tool.power(effect_size=abs(effect), nobs1=n_per_group, alpha=0.05, ratio=1.0))
                power_grid_rows.append(
                    {
                        "group1": ref["gender"],
                        "group2": other["gender"],
                        "n_total": n_total,
                        "n_per_group": n_per_group,
                        "effect_size_h": effect,
                        "power": power,
                    }
                )
            pair_rows.extend(_two_proportion_rows(analysis, "heavy_drinking_30d", ref["gender"], other["gender"]))
    if "ebac_legal" in frame.columns and "alcohol_past12m" in frame.columns and len(gender_summary) >= 2:
        # eBAC over the past-year drinkers, as the R route scopes it
        drinkers = frame.dropna(subset=["gender", "weight", "ebac_legal", "alcohol_past12m"])
        drinkers = drinkers[drinkers["alcohol_past12m"] == 1]
        levels = list(gender_summary["gender"])
        for g2 in levels[1:]:
            pair_rows.extend(_two_proportion_rows(drinkers, "ebac_legal", levels[0], g2))

    overall_prev = float((analysis["heavy_drinking_30d"] * analysis["weight"]).sum() / analysis["weight"].sum())

    power_summary = pd.DataFrame(
        [
            {"metric": "analysis_n", "value": float(len(analysis))},
            {"metric": "heavy_drinking_prevalence_weighted", "value": overall_prev},
            {"metric": "gender_levels_used", "value": float(gender_summary["gender"].nunique())},
        ]
    )

    # ── eBAC endpoint power anchors ──────────────────────────────────
    ebac_endpoints = ["ebac_legal", "ebac_tot"]
    anchor_rows = []
    for ep in ebac_endpoints:
        if ep in analysis.columns:
            ep_prev = (
                float(
                    (analysis[ep].dropna() * analysis.loc[analysis[ep].notna(), "weight"]).sum()
                    / analysis.loc[analysis[ep].notna(), "weight"].sum()
                )
                if analysis[ep].notna().any()
                else 0.0
            )
            for n in sample_sizes:
                try:
                    es = proportion_effectsize(ep_prev, ep_prev * 0.85)
                    pwr = float(power_tool.power(effect_size=abs(es), nobs1=n / 2, alpha=0.05, ratio=1.0))
                except Exception:
                    pwr = float("nan")
                anchor_rows.append({"endpoint": ep, "prevalence": ep_prev, "n_total": n, "power": pwr})

    # ── G*Power reference two-group ──────────────────────────────────
    gpower_rows = []
    for es_label, es_val in [("small", 0.2), ("medium", 0.5), ("large", 0.8)]:
        for n in sample_sizes:
            pwr = float(power_tool.power(effect_size=es_val, nobs1=n / 2, alpha=0.05, ratio=1.0))
            gpower_rows.append({"effect_size_label": es_label, "cohens_h": es_val, "n_total": n, "power": pwr})

    # ── Interaction power analysis ───────────────────────────────────
    interaction_assumptions = []
    interaction_feasibility = []
    interaction_allocations = []
    interaction_imbalance = []
    interaction_pairwise = []
    interaction_targets = []

    if "cannabis_any_use" in analysis.columns and len(gender_summary) >= 2:
        cannabis_prev = float(analysis["cannabis_any_use"].mean())
        for _, grow in gender_summary.iterrows():
            g = grow["gender"]
            g_mask = analysis["gender"] == g
            g_n = int(g_mask.sum())
            g_prev = grow["weighted_prevalence"]
            c_among_g = float(analysis.loc[g_mask, "cannabis_any_use"].mean()) if g_n > 0 else 0.0

            interaction_assumptions.append(
                {
                    "gender": g,
                    "n": g_n,
                    "heavy_drinking_prev": g_prev,
                    "cannabis_prev_in_group": c_among_g,
                    "overall_cannabis_prev": cannabis_prev,
                }
            )

            # Feasibility: need ≥30 per cell for 2×2 interaction
            n_cells = max(1, int(g_n * c_among_g)), max(1, int(g_n * (1 - c_among_g)))
            min_cell = min(n_cells)
            interaction_feasibility.append(
                {
                    "gender": g,
                    "min_cell_n": min_cell,
                    "feasible": "Yes" if min_cell >= 30 else "No",
                    "flag": "" if min_cell >= 30 else "cell_too_small",
                }
            )

            interaction_allocations.append(
                {
                    "gender": g,
                    "n_total": g_n,
                    "n_cannabis": n_cells[0],
                    "n_no_cannabis": n_cells[1],
                    "ratio": round(n_cells[0] / n_cells[1], 3) if n_cells[1] > 0 else float("nan"),
                }
            )

            # Imbalance penalty
            ratio = n_cells[0] / n_cells[1] if n_cells[1] > 0 else 1.0
            penalty = abs(1.0 - ratio) * 0.1
            interaction_imbalance.append(
                {"gender": g, "allocation_ratio": round(ratio, 3), "penalty_factor": round(penalty, 4)}
            )

        # Pairwise interaction power
        genders = gender_summary["gender"].tolist()
        for i in range(len(genders)):
            for j in range(i + 1, len(genders)):
                g1, g2 = genders[i], genders[j]
                p1 = float(gender_summary.loc[gender_summary["gender"] == g1, "weighted_prevalence"].iloc[0])
                p2 = float(gender_summary.loc[gender_summary["gender"] == g2, "weighted_prevalence"].iloc[0])
                diff = abs(p1 - p2)
                es = proportion_effectsize(p1, p2) if p1 > 0 and p2 > 0 else 0.0
                for n in sample_sizes:
                    try:
                        pwr = float(power_tool.power(effect_size=abs(es) * 0.5, nobs1=n / 4, alpha=0.05, ratio=1.0))
                    except Exception:
                        pwr = float("nan")
                    interaction_pairwise.append(
                        {
                            "gender1": g1,
                            "gender2": g2,
                            "prevalence_diff": diff,
                            "interaction_effect_h": round(es * 0.5, 4),
                            "n_total": n,
                            "power": pwr,
                        }
                    )

        # Target sample sizes for 80% power
        for i in range(len(genders)):
            for j in range(i + 1, len(genders)):
                g1, g2 = genders[i], genders[j]
                p1 = float(gender_summary.loc[gender_summary["gender"] == g1, "weighted_prevalence"].iloc[0])
                p2 = float(gender_summary.loc[gender_summary["gender"] == g2, "weighted_prevalence"].iloc[0])
                es = proportion_effectsize(p1, p2) * 0.5 if p1 > 0 and p2 > 0 else 0.0
                try:
                    n80 = (
                        float(power_tool.solve_power(effect_size=abs(es), alpha=0.05, power=0.80, ratio=1.0))
                        if es != 0
                        else float("nan")
                    )
                except Exception:
                    n80 = float("nan")
                interaction_targets.append(
                    {
                        "gender1": g1,
                        "gender2": g2,
                        "target_power": 0.80,
                        "interaction_effect_h": round(es, 4),
                        "required_n_per_group": round(n80, 0) if not np.isnan(n80) else float("nan"),
                        "required_n_total": round(n80 * 4, 0) if not np.isnan(n80) else float("nan"),
                    }
                )

    # ── Randomization schedules ──────────────────────────────────────
    rng = np.random.RandomState(42)
    block_sizes = [4, 6, 8]
    block_rows = []
    for bs in block_sizes:
        n_treated = bs // 2
        n_control = bs - n_treated
        block_rows.append(
            {
                "block_size": bs,
                "n_treated_per_block": n_treated,
                "n_control_per_block": n_control,
                "n_blocks_for_200": 200 // bs,
                "n_blocks_for_1000": 1000 // bs,
            }
        )

    def _make_schedule(name: str, n: int = 200) -> pd.DataFrame:
        schedule = []
        seq_id = 1
        block_id = 1
        while seq_id <= n:
            bs = rng.choice(block_sizes)
            assignments = ["treatment"] * (bs // 2) + ["control"] * (bs - bs // 2)
            rng.shuffle(assignments)
            for a in assignments:
                if seq_id > n:
                    break
                schedule.append({"sequence_id": seq_id, "block_id": block_id, "block_size": bs, "assignment": a})
                seq_id += 1
            block_id += 1
        return pd.DataFrame(schedule)

    outputs = {
        "power_summary": power_summary,
        "power_two_proportion_gender": pd.DataFrame(pair_rows),
        "power_one_proportion_grid": pd.DataFrame(power_grid_rows),
        "power_ebac_endpoint_anchors": pd.DataFrame(anchor_rows),
        "power_gpower_reference_two_group": pd.DataFrame(gpower_rows),
        "power_interaction_assumptions": pd.DataFrame(interaction_assumptions),
        "power_interaction_feasibility_flags": pd.DataFrame(interaction_feasibility),
        "power_interaction_group_allocations": pd.DataFrame(interaction_allocations),
        "power_interaction_imbalance_penalty": pd.DataFrame(interaction_imbalance),
        "power_interaction_pairwise_details": pd.DataFrame(interaction_pairwise),
        "power_interaction_sample_size_targets": pd.DataFrame(interaction_targets),
        "randomization_block_blueprints": pd.DataFrame(block_rows),
        "randomization_schedule_example_heavy_drinking_30d": _make_schedule("heavy_drinking_30d"),
        "randomization_schedule_example_ebac_legal": _make_schedule("ebac_legal"),
        "randomization_schedule_example_ebac_tot": _make_schedule("ebac_tot"),
    }
    return _write_outputs(outputs, output_dir)


def _run_ebac_gender_smote_sensitivity(
    data: pd.DataFrame,
    *,
    treatment: str = "cannabis_any_use",
    binary_outcome: str = "ebac_legal",
    weight_col: str = "weight",
) -> dict[str, pd.DataFrame]:
    """eBAC gender interaction and SMOTE sensitivity analysis."""
    from morie.fn import _glm_core as sm

    from .ml import apply_smote
    from .survey import SurveyDesign

    covariates = ["age_group", "gender", "province_region", "mental_health", "physical_health"]
    required = [binary_outcome, treatment, weight_col, *covariates]
    frame = data.loc[:, [c for c in required if c in data.columns]].dropna().copy()

    design = SurveyDesign(frame, weights_col=weight_col)

    # Gender interaction survey-weighted logistic OR
    formula = f"{binary_outcome} ~ {treatment} + " + " + ".join(covariates) + f" + {treatment}:gender"
    try:
        fit = design.svyglm(formula, family=sm.families.Binomial())
        conf = fit.conf_int()
        or_rows = []
        for term in fit.params.index:
            or_rows.append(
                {
                    "term": term,
                    "log_odds": float(fit.params[term]),
                    "SE": float(fit.bse[term]),
                    "OR": float(np.exp(np.clip(fit.params[term], -700, 700))),
                    "OR_lower95": float(np.exp(np.clip(conf.loc[term, 0], -700, 700))),
                    "OR_upper95": float(np.exp(np.clip(conf.loc[term, 1], -700, 700))),
                    "p_value": float(fit.pvalues[term]),
                }
            )
        gender_or = pd.DataFrame(or_rows)

        # Interaction Wald test
        int_terms = [t for t in fit.params.index if ":" in t and treatment in t and "gender" in t]
        if int_terms:
            contrast = np.zeros((len(int_terms), len(fit.params)))
            idx_list = list(fit.params.index)
            for ri, t in enumerate(int_terms):
                contrast[ri, idx_list.index(t)] = 1.0
            wald = fit.wald_test(contrast, scalar=True)
            from .investigation import _scalarize

            stat = _scalarize(wald.statistic)
            pval = _scalarize(wald.pvalue)
        else:
            stat, pval = 0.0, 1.0

        interaction_tests = pd.DataFrame(
            [
                {
                    "test": f"{treatment}:gender (joint Wald)",
                    "F_stat": stat,
                    "df_num": len(int_terms),
                    "p_value": pval,
                }
            ]
        )
    except Exception:
        gender_or = pd.DataFrame(columns=["term", "log_odds", "SE", "OR", "OR_lower95", "OR_upper95", "p_value"])
        interaction_tests = pd.DataFrame(columns=["test", "F_stat", "df_num", "p_value"])

    # Marginal predicted probabilities by gender
    marginal_rows = []
    for g_level, g_sub in frame.groupby("gender"):
        prev = (
            float((g_sub[binary_outcome] * g_sub[weight_col]).sum() / g_sub[weight_col].sum())
            if g_sub[weight_col].sum() > 0
            else 0.0
        )
        marginal_rows.append({"gender": g_level, "n": len(g_sub), "marginal_prob": prev})
    marginal_probs = pd.DataFrame(marginal_rows)

    # SMOTE sensitivity
    y = frame[binary_outcome].astype(int)
    X = pd.get_dummies(frame[[treatment, *covariates]], drop_first=True, dtype=float)
    X_res, y_res, smote_info = apply_smote(X, y)
    smote_status = pd.DataFrame([smote_info])

    try:
        smote_fit = sm.GLM(y_res, sm.add_constant(X_res), family=sm.families.Binomial()).fit()
        smote_conf = smote_fit.conf_int()
        smote_or = pd.DataFrame(
            {
                "term": smote_fit.params.index,
                "log_odds": smote_fit.params.values,
                "SE": smote_fit.bse.values,
                "OR": np.exp(np.clip(smote_fit.params.values, -700, 700)),
                "OR_lower95": np.exp(np.clip(smote_conf[0].values, -700, 700)),
                "OR_upper95": np.exp(np.clip(smote_conf[1].values, -700, 700)),
                "p_value": smote_fit.pvalues.values,
            }
        )
    except Exception:
        smote_or = pd.DataFrame(columns=["term", "log_odds", "SE", "OR", "OR_lower95", "OR_upper95", "p_value"])

    # SMOTE comparison: original vs SMOTE odds ratios
    compare_rows = []
    for term in gender_or["term"].values:
        orig = gender_or.loc[gender_or["term"] == term]
        resampled = smote_or.loc[smote_or["term"] == term] if term in smote_or["term"].values else pd.DataFrame()
        compare_rows.append(
            {
                "term": term,
                "OR_original": float(orig["OR"].iloc[0]) if len(orig) else float("nan"),
                "OR_smote": float(resampled["OR"].iloc[0]) if len(resampled) else float("nan"),
                "p_original": float(orig["p_value"].iloc[0]) if len(orig) else float("nan"),
                "p_smote": float(resampled["p_value"].iloc[0]) if len(resampled) else float("nan"),
            }
        )
    smote_compare = pd.DataFrame(compare_rows)

    return {
        "ebac_gender_interaction_svy_or": gender_or,
        "ebac_gender_interaction_tests": interaction_tests,
        "ebac_gender_marginal_probs": marginal_probs,
        "ebac_smote_status": smote_status,
        "ebac_smote_or": smote_or,
        "ebac_smote_compare": smote_compare,
    }


def _load_dataset_frame(
    dataset_key: str | None = None,
    cpads_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load a dataset by key, path, or from the DB.

    Supports three modes:
    1. **Catalog key** -- e.g. "ocp21", "hibp" -> loads from SQLite DB
    2. **Arbitrary CSV/XLSX path** -- e.g. "/tmp/my_data.csv" -> reads directly
    3. **Default** -- CPADS CSV (backward compat when no key/path given)

    Parameters
    ----------
    dataset_key : str, optional
        Short key from DATASET_CATALOG, OR a file path to any CSV/XLSX.
        When it's a file path, the data is loaded directly without
        column validation (dataset-agnostic mode).
    cpads_csv : str or Path, optional
        Legacy: direct path to a CPADS CSV file.
    """
    if dataset_key:
        # Check if it's a file path first
        key_path = Path(dataset_key)
        if key_path.suffix in {".csv", ".xlsx", ".xls", ".tsv", ".parquet"} or key_path.exists():
            if key_path.exists():
                if key_path.suffix == ".csv":
                    return pd.read_csv(key_path)
                elif key_path.suffix == ".tsv":
                    return pd.read_csv(key_path, sep="\t")
                elif key_path.suffix in {".xlsx", ".xls"}:
                    return pd.read_excel(key_path)
                elif key_path.suffix == ".parquet":
                    return pd.read_parquet(key_path)
            else:
                raise FileNotFoundError(f"Dataset file not found: {dataset_key}")

        # Check if it's a catalog key
        from .data import DATASET_CATALOG, load_dataset

        if dataset_key in DATASET_CATALOG:
            return load_dataset(dataset_key)

        # Not a known key and not a file path -- try fuzzy match
        matches = [k for k in DATASET_CATALOG if dataset_key.lower() in k]
        if matches:
            return load_dataset(matches[0])

        raise ValueError(
            f"Unknown dataset: {dataset_key}. "
            f"Pass a catalog key ({', '.join(sorted(list(DATASET_CATALOG)[:5]))}...) "
            f"or a file path (CSV/XLSX/TSV/Parquet)."
        )

    # Default: CPADS via CSV (backward compat)
    csv_path = cpads_csv or DEFAULT_CPADS_CSV
    return load_cpads_analysis_data(csv_path)


# Modules with BOTH an R and a Python implementation; only these may fall
# back to Python when the R stage fails (with a logged warning).
_PY_FALLBACK_MODULES = frozenset(
    {
        "power-design",
        "logistic-models",
        "model-comparison",
        "propensity-scores",
        "treatment-effects",
        "ebac-selection-adjustment-ipw",
        "ebac-gender-smote-sensitivity",
    }
)


_R_INSTALL_HINT = (
    "install R, then `morie r-install` (rmorie from r-universe, prebuilt on macOS and Windows) and run the module again"
)


_R_READY: bool | None = None
_R_PACKAGE: str | None = None  # the R package the bridge loads: rmorie or morie, whichever matches this version


def _sp():
    """The program launcher (``morie._launch``, interactive layer); the R route needs it."""
    from ._interactive import launcher

    return launcher("Running R-backed modules")


def _r_stage_error(exc: BaseException) -> bool:
    """RuntimeError, OSError, or an error from the program launcher: the R stage itself failed."""
    if isinstance(exc, RuntimeError | OSError):
        return True
    try:
        from . import _launch
    except ImportError:  # no launcher, so it cannot have raised this
        return False
    return isinstance(exc, _launch.SubprocessError)


def _r_mismatch_mode() -> str:
    """warn, quiet or strict; MORIE_ALLOW_R_VERSION_MISMATCH (the older switch) still means quiet."""
    if os.environ.get("MORIE_ALLOW_R_VERSION_MISMATCH"):
        return "quiet"
    try:
        from .llm_config import r_mismatch_mode

        return r_mismatch_mode()
    except Exception:  # noqa: BLE001 - an unreadable settings file must not stop an analysis
        return "warn"


_R_VERSION_MATCH: bool | None = None  # the R package in use has this morie's version
_R_MISMATCH_NOTE: str | None = None  # the warning for another version, shown once
_R_WARNED = False


def _r_route_ready(*, warn: bool = True) -> None:
    """Raise at once when Rscript or a matching R package (rmorie or morie) is missing; cached per process.

    Both R arms carry the modules. The one whose version equals this morie's is used (rmorie when both
    match). When only another version is installed, the ``r.mismatch`` setting decides: warn (the
    default) uses it and says so once, quiet uses it silently, strict refuses. Set it with
    ``morie config set r.mismatch ...`` or ``MORIE_R_VERSION_MISMATCH``.
    """
    global _R_READY, _R_PACKAGE, _R_VERSION_MATCH, _R_MISMATCH_NOTE, _R_WARNED
    if _R_READY:
        if warn and _R_MISMATCH_NOTE and not _R_WARNED:
            import warnings

            _R_WARNED = True
            warnings.warn(_R_MISMATCH_NOTE, RuntimeWarning, stacklevel=2)
        return
    rscript = _rscript_bin()
    if rscript is None:
        raise RuntimeError("Rscript is not available on PATH.")
    sp = _sp()
    probe = sp.r_expr(
        'for (p in c("rmorie", "morie")) if (requireNamespace(p, quietly = TRUE)) '
        'cat(p, as.character(utils::packageVersion(p)), "\\n")',
        rscript=rscript,
        options=("--vanilla",),
        stdin=sp.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    found = [ln.split()[:2] for ln in probe.stdout.splitlines() if len(ln.split()) >= 2]
    if not found:
        raise RuntimeError("No R package for the R-backed modules is installed (install rmorie, or morie's R package)")
    from . import __version__ as py_version

    match = next((pkg for pkg, ver in found if ver == py_version), None)
    if match is None:
        match = found[0][0]
        have = ", ".join(f"{pkg} {ver}" for pkg, ver in found)
        mode = _r_mismatch_mode()
        dev = py_version.startswith("0.0.0") or "+" in py_version  # a source checkout has no release version
        if mode == "strict":
            raise RuntimeError(
                f"R {have} {'is' if len(found) == 1 else 'are'} installed, but this is morie {py_version}, and "
                f"r.mismatch is strict (run `morie r-install` for the same version, or "
                f"`morie config set r.mismatch warn` to use the installed one)"
            )
        if mode == "warn" and not dev:
            _R_MISMATCH_NOTE = (
                f"using R {match} {dict(found)[match]} with morie {py_version}: results may differ from "
                f"morie {py_version}'s R code (`morie r-install` installs the same version; "
                f"`morie config set r.mismatch quiet` hides this)"
            )
    _R_PACKAGE = match
    _R_VERSION_MATCH = any(pkg == match and ver == py_version for pkg, ver in found)
    _R_READY = True
    if warn and _R_MISMATCH_NOTE:
        import warnings

        _R_WARNED = True
        warnings.warn(_R_MISMATCH_NOTE, RuntimeWarning, stacklevel=2)


def _r_package_absent(exc: BaseException) -> bool:
    """True when the R route failed only because R or its package is not installed."""
    text = str(exc)
    return (
        "Rscript is not available" in text
        or "No R package for the R-backed modules is installed" in text
        or "there is no package called" in text
    )


def r_route_problem(module_name: str, exc: BaseException) -> str:
    """One line for the terminal when an R-backed module cannot run."""
    if "the R-backed modules need the same version" in str(exc):
        return f"{module_name}: {exc}"
    if _r_package_absent(exc):
        hint = (
            "run `morie r-install` (rmorie from r-universe, prebuilt on macOS and Windows) and run the module again"
            if _rscript_bin()
            else _R_INSTALL_HINT
        )
        return (
            f"{module_name} runs in R and this machine has no R package for it: {hint}. "
            f"The Python-only modules are: {', '.join(sorted(_PY_FALLBACK_MODULES))}."
        )
    text = str(exc)
    tail = ""
    if "STDERR:" in text:
        err_lines = [ln for ln in text.split("STDERR:", 1)[1].splitlines() if ln.strip()]
        tail = f": {err_lines[-1].strip()}" if err_lines else ""
    return f"{module_name}: the R-backed run failed{tail} (`morie doctor` checks the R side)"


# directories made by _cpads_csv_for_run, the only ones run_module may remove
_STAGED_DIRS: set[Path] = set()


def _cpads_csv_for_run(cpads_csv: str | Path, dataset_key: str | None) -> str | Path:
    """The CSV the module stages (R bridge or Python) will read.

    The R bridge takes a path only, so a ``dataset_key`` and the real PUMF
    pulled into the Python store (``morie pull ocp21``) are written to a
    CSV first; otherwise ``--dataset`` was dropped on the R route and a
    pulled PUMF stayed invisible to it (both arms then ran the synthetic
    frame while the user believed otherwise).
    """
    import tempfile

    from .data import cached_cpads, load_dataset

    frame = None
    label = ""
    if dataset_key:
        frame = load_dataset(dataset_key)
        label = dataset_key
    elif _is_synthetic_cpads_path(Path(cpads_csv).expanduser().resolve()):
        frame = cached_cpads()
        label = "ocp21-cached"
    if frame is None:
        return cpads_csv
    # a private directory (mode 0700), removed when the run ends: a fixed name in the shared
    # temp dir let runs overwrite each other's input and left the copy behind
    staging = Path(tempfile.mkdtemp(prefix="morie-dataset-"))
    _STAGED_DIRS.add(staging)
    dest = staging / f"{label.replace('/', '__')}.csv"
    frame.to_csv(dest, index=False)
    return dest


def run_module(
    module_name: str,
    *,
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    dataset_key: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run one implemented MORIE module.

    Parameters
    ----------
    module_name : str
        Name from MODULE_SPECS (e.g. "power-design").
    cpads_csv : str or Path
        Legacy: direct path to CPADS CSV.
    dataset_key : str, optional
        Short key from DATASET_CATALOG. When provided, loads from DB
        instead of the CSV path. Default: None (uses cpads_csv).
    output_dir : str or Path, optional
        Directory for CSV outputs.
    """
    if module_name not in MODULE_SPECS:
        valid = ", ".join(sorted(MODULE_SPECS))
        raise ValueError(f"Unknown module: {module_name}. Valid modules: {valid}")

    if dataset_key:
        # a key that names nothing is the user's mistake, reported before anything about R
        from .data import _fuzzy_match_key
        from .datahub import is_hosted_key

        if _fuzzy_match_key(dataset_key) is None and not is_hosted_key(dataset_key):
            raise KeyError(f"Unknown dataset key: {dataset_key!r} (morie list-datasets shows the keys)")
    if module_name not in _PY_FALLBACK_MODULES:
        _r_route_ready()  # R and its package first: loading the frame took 10-14 s before this failed
    staged = _cpads_csv_for_run(cpads_csv, dataset_key)
    try:
        return _run_module_on(module_name, staged, dataset_key, output_dir)
    finally:
        # only a directory _cpads_csv_for_run made itself: never a caller's (a path's parent can be ".")
        staging = Path(staged).parent
        if staging in _STAGED_DIRS:
            _STAGED_DIRS.discard(staging)
            from ._safe_rm import rmtree_owned

            rmtree_owned(staging, owned=True)


def _run_module_on(module_name: str, cpads_csv, dataset_key, output_dir) -> dict[str, object]:
    """``run_module`` once its input CSV is staged."""
    try:
        # a Python-capable module takes the Python route (this morie's own code) rather than run
        # another version of the R arm; an R-only module uses the installed R package (r.mismatch)
        python_capable = module_name in _PY_FALLBACK_MODULES
        _r_route_ready(warn=not python_capable)
        if python_capable and not _R_VERSION_MATCH:
            raise RuntimeError("the R-backed modules need the same version")
        return _run_r_module(module_name, cpads_csv=cpads_csv, output_dir=output_dir)
    except Exception as exc:
        if not _r_stage_error(exc):
            raise
        # Only these exception classes mean "the R stage itself failed"
        # (missing R, bridge error, R-side error). Anything else is a bug
        # in OUR code and must not be masked by the Python fallback.
        if module_name not in _PY_FALLBACK_MODULES:
            raise
        if _r_package_absent(exc) or "the R-backed modules need the same version" in str(exc):
            logging.getLogger(__name__).info(
                "%s: no R package of this version is installed; using the Python implementation", module_name
            )
        else:
            logging.getLogger(__name__).warning(
                "R implementation of %s failed (%s); falling back to the "
                "Python implementation. An R-side regression would otherwise "
                "be invisible - investigate if unexpected.",
                module_name,
                str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
            )

    if module_name == "power-design":
        return run_power_design_module(cpads_csv, output_dir=output_dir)

    frame = _load_dataset_frame(dataset_key=dataset_key, cpads_csv=cpads_csv)
    if module_name == "logistic-models":
        outputs = run_weighted_logistic_analysis(frame)
    elif module_name == "model-comparison":
        outputs = compare_nested_logistic_models(frame)
    elif module_name == "propensity-scores":
        outputs = run_propensity_ipw_analysis(frame)
    elif module_name == "treatment-effects":
        outputs = run_treatment_effects_analysis(frame)
    elif module_name == "ebac-selection-adjustment-ipw":
        outputs = run_ebac_selection_ipw_analysis(frame)
    elif module_name == "ebac-gender-smote-sensitivity":
        outputs = _run_ebac_gender_smote_sensitivity(frame)
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled module: {module_name}")

    csv_outputs = {
        name: table for name, table in outputs.items() if isinstance(table, pd.DataFrame) and name != "analysis_frame"
    }
    return _write_outputs(csv_outputs, output_dir)


def run_modules(
    module_names: list[str] | None = None,
    *,
    cpads_csv: str | Path = DEFAULT_CPADS_CSV,
    dataset_key: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run multiple implemented modules and return their output tables."""
    module_names = module_names or list(MODULE_SPECS)
    return {
        name: run_module(
            name,
            cpads_csv=cpads_csv,
            dataset_key=dataset_key,
            output_dir=output_dir,
        )
        for name in module_names
    }
