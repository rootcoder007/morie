from morie.fn import _frame_core as pd
import pytest

from morie.causal import (
    effective_sample_size,
    run_ebac_selection_ipw_analysis,
    run_propensity_ipw_analysis,
)


def test_run_propensity_ipw_analysis_returns_expected_tables():
    frame = pd.DataFrame(
        {
            "cannabis_any_use": [1, 1, 0, 0, 1, 0],
            "heavy_drinking_30d": [1, 0, 0, 1, 1, 0],
            "age_group": [1, 2, 1, 2, 3, 4],
            "gender": [0, 1, 0, 1, 0, 1],
            "province_region": [0, 1, 0, 1, 0, 1],
            "mental_health": [1, 2, 2, 3, 4, 4],
            "physical_health": [1, 2, 2, 3, 4, 4],
            "weight": [1.0, 1.2, 0.8, 1.1, 1.3, 0.9],
        }
    )

    result = run_propensity_ipw_analysis(frame)

    assert "ipw_results" in result
    assert list(result["ipw_results"]["estimand"]) == ["ATE"]
    assert "ess_ipw_trimmed" in set(result["diagnostics"]["metric"])


@pytest.mark.filterwarnings("ignore:divide by zero encountered in scalar divide:RuntimeWarning")
def test_run_ebac_selection_ipw_analysis_returns_key_outputs():
    # The model carries an intercept, the treatment and five covariates:
    # seven parameters. The old eight-row fixture dropped to six complete
    # rows, so the fit was rank deficient and only "passed" because
    # statsmodels returned NaN coefficients instead of refusing. The
    # native fitter refuses, correctly, so the fixture now carries enough
    # rows to identify the model.
    n = 48
    rows = {
        "weight": [0.8 + 0.02 * (i % 25) for i in range(n)],
        "alcohol_past12m": [1] * n,
        "ebac_tot": [None if i % 11 == 0 else 0.02 + 0.005 * (i % 19)
                     for i in range(n)],
        "ebac_legal": [(i * 7) % 3 == 0 for i in range(n)],
        "cannabis_any_use": [(i * 5) % 4 == 0 for i in range(n)],
        "age_group": [1 + (i % 4) for i in range(n)],
        "gender": [i % 2 for i in range(n)],
        "province_region": [i % 3 for i in range(n)],
        "mental_health": [1 + (i % 5) for i in range(n)],
        "physical_health": [1 + ((i + 2) % 5) for i in range(n)],
        "heavy_drinking_30d": [(i * 3) % 5 == 0 for i in range(n)],
    }
    rows["ebac_legal"] = [int(v) for v in rows["ebac_legal"]]
    rows["cannabis_any_use"] = [int(v) for v in rows["cannabis_any_use"]]
    rows["heavy_drinking_30d"] = [int(v) for v in rows["heavy_drinking_30d"]]
    frame = pd.DataFrame(rows)

    result = run_ebac_selection_ipw_analysis(frame)

    assert "ebac_final_ipw_or" in result
    assert "ebac_legal_or_cannabis" in set(
        result["ebac_final_ipw_comparison"]["metric"])
    # The odds ratio and its interval have to be positive and finite, or
    # the fit silently degenerated again the way the old fixture did.
    or_tbl = result["ebac_final_ipw_or"]
    ors = [float(v) for v in or_tbl["or"]]
    lo = [float(v) for v in or_tbl["or_lower95"]]
    hi = [float(v) for v in or_tbl["or_upper95"]]
    assert ors, "no odds ratios reported"
    for o, a, b in zip(ors, lo, hi):
        assert o == o and 0 < o < float("inf")
        assert a == a and b == b
        assert a <= o <= b


def test_effective_sample_size_positive():
    assert effective_sample_size(pd.Series([1.0, 2.0, 3.0])) > 0
