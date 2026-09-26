"""mrm_diagnostics: balance threshold and glm-equivalent propensity scores."""


def test_balance_threshold_and_glm_propensity():
    import math

    import pandas as pd

    from morie import mrm_diagnostics as M

    rows = []
    for i in range(1, 121):
        x1, x2 = math.sin(i), math.cos(1.3 * i)
        d = int(0.8 * x1 - 0.5 * x2 + 0.6 * math.sin(2.7 * i) > 0)
        rows.append({"x1": x1, "x2": x2, "d": d, "y": 1 + 0.7 * d + x1 + 0.4 * math.cos(3.3 * i)})
    df = pd.DataFrame(rows)
    # |SMD| = 159.33% (x1) and 85.99% (x2)
    assert M.mrm_check_balancing(df, treatment_col="d", covariates=["x1", "x2"], threshold_pct=100).n_imbalanced == 1
    assert M.mrm_check_balancing(df, treatment_col="d", covariates=["x1", "x2"], threshold_pct=200).n_imbalanced == 0
    # reference: R glm(d ~ x1 + x2, binomial) fitted values (the R arm)
    ov = M.mrm_check_overlap(df, treatment_col="d", covariates=["x1", "x2"])
    assert abs(ov.e_treated_quantiles["q2.5"] - 0.1936569839) < 1e-9
    assert abs(ov.e_treated_quantiles["q97.5"] - 0.997076407) < 1e-9
    assert (ov.common_support_lower, ov.common_support_upper) == (0.1658, 0.9031)
    me = M.mrm_median_causal_effect(df, treatment_col="d", outcome_col="y", covariates=["x1", "x2"])
    assert me.n_matched == 57
    assert me.median_treatment_effect == 2.0311
