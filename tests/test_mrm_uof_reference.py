"""mrm_uof against the R arm (Gini, Hill alpha, locality chi-square, Wilson)."""

import math

from morie import mrm_uof as U
from morie.fn import _frame_core as pd


def _df():
    rows = []
    for i in range(300):
        rows.append(
            {
                "force": f"F{(i * 7) % 13}",
                "weapon": ["taser", "baton", "firearm", "spray"][(i * 3) % 4],
                "year": 2018 + i % 6,
                "region_at": ["N", "S", "E", "W"][i % 4],
                "region_now": ["N", "S", "E", "W"][(i // 3) % 4],
                "race": ["A", "B", "C"][(i * 5) % 3],
                "injury": int(math.sin(i * 1.7) > 0.2),
            }
        )
    return pd.DataFrame(rows)


def test_uof_statistics_match_r_arm():
    df = _df()
    fc = U.mrm_uof_force_concentration(df, "force").payload
    # R arm: gini 0.003076923, alpha 1.318596
    assert abs(fc["gini"] - 0.04 / 13) < 1e-12
    assert abs(fc["pareto_alpha_mle"] - 1.3185963376937009) < 1e-12
    rl = U.mrm_uof_region_locality(df, "region_at", "region_now").payload
    assert (rl["chi2"], rl["diagonal_share"]) == (100.0, 1 / 3)
    dd = U.mrm_uof_demographic_disparity(df, "race", "injury").payload
    a = next(c for c in dd["per_category"] if c["category"] == "A")
    # uncorrected Wilson, as the R arm's .uof_wilson_ci
    assert (a["n"], a["k"]) == (100, 44)
    assert abs(a["lo"] - 0.346720266301608) < 1e-12
    assert abs(a["hi"] - 0.53771895348323) < 1e-12


def test_wilson_corners():
    assert U._wilson_ci(0, 10)[0] == 0.0
    assert abs(U._wilson_ci(10, 10)[1] - 1.0) < 1e-15
    assert all(math.isnan(v) for v in U._wilson_ci(0, 0))
