"""mrm_design: Tukey HSD and IPW against the R arm (TukeyHSD, glm)."""

import math

import pandas as pd

from morie import mrm_design as M


def test_tukey_hsd_matches_r_tukeyhsd():
    # reference: TukeyHSD(aov(y ~ g)) in R
    g = ["a"] * 12 + ["b"] * 15 + ["c"] * 18
    y = [{"a": 0, "b": 0.8, "c": 1.1}[v] + math.sin((i + 1) * 1.7) for i, v in enumerate(g)]
    t = M.mrm_anova_oneway(pd.DataFrame({"g": g, "y": y}), response_col="y", group_col="g").tukey_hsd
    assert list(t["pair"]) == ["b-a", "c-a", "c-b"]
    ref = [
        (0.731056551708389, 0.0371281905107372, 1.42498491290604, 0.0369767876305993),
        (1.00181212947578, 0.334079252582816, 1.66954500636875, 0.00206866382933946),
        (0.270755577767394, -0.355633384058808, 0.897144539593596, 0.549915670968965),
    ]
    for row, r in zip(zip(t["diff"], t["lwr"], t["upr"], t["p adj"]), ref):
        for a, b in zip(row, r):
            assert abs(a - b) < 1e-9


def test_causal_design_ipw_uses_the_glm_propensity():
    rows = []
    for i in range(1, 151):
        x1, x2 = math.sin(i), math.cos(i * 1.4)
        d = int(0.7 * x1 + 0.4 * math.sin(i * 3.1) > 0)
        rows.append({"x1": x1, "x2": x2, "D": d, "Y": 1 + 0.5 * d + x1 + 0.3 * math.cos(i * 2.2)})
    r = M.mrm_causal_design(pd.DataFrame(rows), treatment_col="D", outcome_col="Y", covariates=["x1", "x2"])
    # R arm: glm(D ~ x1 + x2, binomial) weights give 1.036702
    assert r.estimate == 1.036702
