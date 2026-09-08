"""Anchored tests for pelt (Killick-Fearnhead-Eckley 2012)."""

from morie.fn.pelt import pelt

X1 = [0.1, -0.2, 0.05, 0.3, -0.1, 5.2, 4.9, 5.1, 5.3, 4.8,
      1.9, 2.1, 2.0, 1.8, 2.2]


def test_pelt_matches_changepoint_package_anchor():
    # Anchor: changepoint::cpt.mean(x1, method="PELT",
    # penalty="Manual", pen.value=3.0, test.stat="Normal") returns
    # cpts {5, 10} (run 2026-08-09, changepoint 2.x, R 4.6.1).
    r = pelt(X1, "mean", 3.0)
    assert r["changepoints"] == [5, 10]
    # and cpt.meanvar with pen.value=6.0 also returns {5, 10}
    r = pelt(X1, "meanvar", 6.0)
    assert r["changepoints"] == [5, 10]


def test_pelt_objective_arithmetic():
    # Independent arithmetic: F(n) must equal
    # sum_segments ssdev + beta * (m + 1) - beta ... eq (3) of the
    # paper counts beta once per segment and F(0) = -beta, so
    # F(n) = sum C + (m + 1) beta - beta = sum C + m beta.
    r = pelt(X1, "mean", 3.0)
    segs = [(0, 5), (5, 10), (10, 15)]
    tot = 0.0
    for a, b in segs:
        seg = X1[a:b]
        mu = sum(seg) / len(seg)
        tot += sum((v - mu) ** 2 for v in seg)
    m = 2
    assert abs(r["objective"] - (tot + m * 3.0)) < 1e-12


def test_pelt_no_change_on_constantish_series():
    r = pelt([1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 1.0, 1.01], "mean", 5.0)
    assert r["changepoints"] == []


def test_pelt_penalty_monotone():
    lo = pelt(X1, "mean", 0.5)
    hi = pelt(X1, "mean", 100.0)
    assert lo["n_changepoints"] >= hi["n_changepoints"]
    assert hi["changepoints"] == []
