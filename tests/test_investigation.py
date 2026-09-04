from morie.fn import _frame_core as pd

from morie.investigation import (
    compare_nested_logistic_models,
    run_treatment_effects_analysis,
    run_weighted_logistic_analysis,
)


def _draws(n, seed=20260903):
    """Deterministic uniforms from a pinned LCG (Numerical Recipes).

    Hand-written cyclic patterns (i % 3, i % 4, ...) look like data but
    make one column an exact function of another -- the previous fixture
    had the outcome equal to 1 exactly when province_region was 0, which
    is perfect separation, and the fit was correctly refused as singular.
    Independent pseudo-random draws avoid that.
    """
    s = seed
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) % (2 ** 32)
        out.append(s / 2 ** 32)
    return out


def _pick(u, values):
    return [values[int(v * len(values)) % len(values)] for v in u]


def _frame():
    n = 120
    u = _draws(n * 8)
    c = [u[k * n:(k + 1) * n] for k in range(8)]
    return pd.DataFrame(
        {
            "weight": [0.6 + 1.0 * v for v in c[0]],
            "alcohol_past12m": [1] * n,
            "heavy_drinking_30d": [int(v < 0.45) for v in c[1]],
            "ebac_tot": [0.20 * v for v in c[2]],
            "ebac_legal": [int(v < 0.40) for v in c[3]],
            "cannabis_any_use": [int(v < 0.35) for v in c[4]],
            "age_group": _pick(c[5], [1, 2, 3, 4]),
            "gender": [int(v < 0.5) for v in c[6]],
            "province_region": _pick(c[7], [0, 1, 2]),
            "mental_health": _pick(c[0], [1, 2, 3, 4, 5]),
            "physical_health": _pick(c[1], [1, 2, 3, 4, 5]),
        }
    )


def test_run_weighted_logistic_analysis_outputs_expected_tables():
    outputs = run_weighted_logistic_analysis(_frame())
    assert "logistic_odds_ratios" in outputs
    assert "logistic_interaction_tests" in outputs
    assert "OR" in outputs["logistic_odds_ratios"].columns


def test_compare_nested_logistic_models_outputs_summary():
    outputs = compare_nested_logistic_models(_frame())
    assert "model_comparison_summary" in outputs
    assert "model_comparison_wald_tests" in outputs
    assert set(outputs["model_comparison_summary"]["model"]) >= {"Model 0", "Model 1", "Model 2", "Model 3", "Model 4"}


def test_run_treatment_effects_analysis_outputs_summary_and_cate():
    outputs = run_treatment_effects_analysis(_frame())
    assert "treatment_effects_summary" in outputs
    assert "cate_subgroup_estimates" in outputs
    assert set(outputs["treatment_effects_summary"]["estimand"]) >= {"ATE", "ATT", "ATC"}
