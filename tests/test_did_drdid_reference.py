"""Callaway-Sant'Anna ATT(g, t) and Sant'Anna-Zhao DR-DiD against R.

References: R ``did::att_gt(y, t, id, g, xformla = ~x, est_method = m,
control_group = "nevertreated", bstrap = FALSE, base_period = "varying")``
and ``DRDID::drdid_rc1(y, post, D, cbind(1, x1, x2), boot = FALSE)`` on the
deterministic data built below (never-treated coded 0 for R, inf here).
"""

import math

import pytest

from morie import did
from morie.fn import _frame_core as pd

REF = {
    "doubly_robust": {
        (3, 3): 0.94155496227272473,
        (3, 4): 1.4724024110277192,
        (3, 5): 2.0473507615710287,
        (4, 4): 1.0172858083099621,
        (4, 5): 1.5537602811443496,
    },
    "ipw": {
        (3, 3): 0.93955404081117877,
        (3, 4): 1.4684614565968612,
        (3, 5): 2.0415165656129477,
        (4, 4): 1.0183336593405952,
        (4, 5): 1.5558307102017066,
    },
    "outcome_regression": {
        (3, 3): 0.94168154049951425,
        (3, 4): 1.476285934019536,
        (3, 5): 2.0531176629837606,
        (4, 4): 1.0182811057409895,
        (4, 5): 1.5561974826054348,
    },
}


def _panel():
    rows = []
    for i in range(300):
        x = math.sin(1.7 * i) + 0.5 * math.cos(0.3 * i)
        g = [3, 4, float("inf")][i % 3] if (i * 7) % 5 != 0 else (3 if x > 0.3 else float("inf"))
        for t in range(1, 6):
            eff = (1.0 + 0.5 * (t - g)) if t >= g else 0.0
            y = 2.0 + 0.3 * t + 1.1 * x + 0.4 * x * t + eff + 0.8 * math.sin(3.1 * i + 1.3 * t)
            rows.append({"id": i, "t": t, "y": y, "g": g, "x": x})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", sorted(REF))
def test_group_time_att_matches_did_att_gt(method):
    r = did.group_time_att(_panel(), "y", "id", "t", "g", covariates=["x"], method=method, n_bootstrap=2)
    got = {(int(a), int(b)): v for a, b, v in zip(r["cohort"], r["time"], r["att"])}
    assert set(got) == set(REF[method])
    for k, ref in REF[method].items():
        assert abs(got[k] - ref) <= 1e-10 * abs(ref)


def test_doubly_robust_is_not_the_outcome_regression():
    # the DR correction used to be multiplied by 0, returning the OR estimate
    r = did.group_time_att(_panel(), "y", "id", "t", "g", covariates=["x"], method="doubly_robust", n_bootstrap=2)
    assert abs(r["att"][0] - REF["outcome_regression"][(3, 3)]) > 1e-4


def test_did_doubly_robust_matches_drdid_rc1():
    rows = []
    for i in range(600):
        x1 = math.sin(1.3 * i)
        x2 = math.cos(0.7 * i + 0.4)
        d = 1 if (math.sin(2.1 * i) + 0.6 * x1) > 0.1 else 0
        post = i % 2
        y = 1.0 + 0.5 * x1 - 0.8 * x2 + 0.7 * post + 0.4 * d + 1.3 * d * post + 0.9 * math.sin(5.3 * i)
        rows.append({"y": y, "d": d, "post": post, "x1": x1, "x2": x2})
    r = did.did_doubly_robust(pd.DataFrame(rows), "y", "d", "post", ["x1", "x2"], n_bootstrap=2)
    assert abs(r.estimate - 1.3769830934895855) <= 1e-10


def test_twfe_and_event_study_match_fixest():
    # feols(y ~ D | unit + time, cluster = ~unit); event study with binned
    # endpoints and wald() on the pre-period coefficients; unbalanced panel
    import math

    from morie import did as D
    from morie.fn import _frame_core as pd

    rows = []
    for unit in range(1, 41):
        for time in range(1, 9):
            g = 4 if unit <= 12 else (6 if unit <= 24 else 0)
            dd = int(g > 0 and time >= g)
            x = round(math.sin(1.3 * unit) + 0.2 * time, 4)
            y = round(
                0.5 * unit / 10
                + 0.3 * time
                + 1.5 * dd
                + 0.4 * dd * (time - g) * (g > 0)
                + 0.3 * math.sin(2.7 * unit * time)
                + 0.2 * x,
                5,
            )
            rows.append((unit, time, g, dd, y))
    full = {k: [r[j] for r in rows] for j, k in enumerate(("unit", "time", "g", "D", "y"))}
    d = pd.DataFrame(full)
    r = D.did_panel_fe(d, "y", "D", "unit", "time", cluster="unit")
    assert abs(r.estimate - 1.88383260684) <= 1e-10 and abs(r.std_error - 0.0758991745971) <= 1e-11
    keep = [i for i, r_ in enumerate(rows) if not (r_[0] in (3, 17, 30) and r_[1] in (2, 5))]
    u = pd.DataFrame({k: [v[i] for i in keep] for k, v in full.items()})
    r = D.did_panel_fe(u, "y", "D", "unit", "time", cluster="unit")
    assert abs(r.estimate - 1.8824137808) <= 1e-10 and abs(r.std_error - 0.0767819050896) <= 1e-11
    u["tt"] = [float(v) if v > 0 else float("nan") for v in u["g"].tolist()]
    es = D.event_study(u, "y", "unit", "time", "tt", cluster="unit")
    c = es.coefficients
    rt = c["relative_time"].tolist()
    for k, b, s in (
        (-4, -0.0579744164321, 0.0940534922065),
        (0, 1.43679248031, 0.0899078370266),
        (4, 3.07422506574, 0.0939645477063),
    ):
        i = rt.index(k)
        assert abs(c["estimate"].tolist()[i] - b) <= 1e-10 and abs(c["std_error"].tolist()[i] - s) <= 1e-11
    assert abs(es.pre_trend_f_stat - 0.216136594631) <= 1e-10
