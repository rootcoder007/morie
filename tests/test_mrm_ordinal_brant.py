"""Threshold-specific ordinal logit: Brant's test of proportional odds."""

import math

from morie.fn import _frame_core as pd
from morie.fn import _stats_core as stats
from morie.mrm_primitives.ordinal import threshold_specific_ordinal


def test_brant_test_matches_r_arm():
    rows = []
    for i in range(1, 501):
        x1, x2 = math.sin(i * 1.3), math.cos(i * 0.7)
        lat = 0.8 * x1 - 0.5 * x2 + 0.9 * math.sin(i * 2.9) + 0.4 * x1 * (math.sin(i * 5.1) > 0)
        y = 0 if lat <= -0.6 else 1 if lat <= 0 else 2 if lat <= 0.7 else 3
        rows.append({"y": "abcd"[y], "x1": x1, "x2": x2})
    r = threshold_specific_ordinal(
        pd.DataFrame(rows), outcome_col="y", covariate_cols=["x1", "x2"], ordinal_levels=list("abcd")
    )
    # R arm .mrm_brant_test: 3.6205578140454 (brant::brant's 3.61332812 has an
    # untransposed off-diagonal covariance block)
    assert abs(r.proportional_odds_stat - 3.6205578140454) < 1e-8
    assert r.proportional_odds_df == 4
    assert abs(r.proportional_odds_p - float(stats.chi2.sf(r.proportional_odds_stat, 4))) < 1e-14
