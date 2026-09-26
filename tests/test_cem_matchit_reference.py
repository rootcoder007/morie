"""Coarsened exact matching against R MatchIt.

Reference: matchit(d ~ x1 + x2, method = "cem", estimand = "ATT",
cutpoints = list(x1 = "q5", x2 = "q5")) on the deterministic data below:
240 matched units, control weights summing to 115, and the weights of the
first three matched controls (ids 0, 2, 6).
"""

import math

from morie import matching
from morie.fn import _frame_core as pd


def test_cem_matches_matchit():
    rows = []
    for i in range(300):
        x1 = math.sin(1.7 * i) * 3 + i * 0.001
        x2 = math.cos(0.9 * i + 0.2) * 2 + i * 0.0007
        d = 1 if (math.sin(3.1 * i) + 0.4 * x1) > 0.3 else 0
        rows.append({"id": i, "d": d, "x1": x1, "x2": x2})
    m = matching.match_cem(pd.DataFrame(rows), "d", ["x1", "x2"], n_bins=5)
    md = m.matched_data
    w = {int(i): float(v) for i, v in zip(md["id"], md["_cem_weight"])}
    dd = {int(i): int(v) for i, v in zip(md["id"], md["d"])}
    assert len(w) == 240
    assert abs(sum(v for k, v in w.items() if dd[k] == 0) - 115.0) <= 1e-12
    for k, ref in ((0, 0.78857142857142859), (2, 0.30666666666666664), (6, 0.16727272727272727)):
        assert abs(w[k] - ref) <= 1e-14
