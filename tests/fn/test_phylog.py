"""Anchored tests for phylog (TempEst root-to-tip regression).

Anchors: an exact strict-clock construction (d = u (t - tr) with
known u and tr must be recovered exactly), and hand OLS arithmetic.
"""

from morie.fn.phylog import phylog, phylogenetic_dating


def test_phylog_exact_clock_recovery():
    u, tr = 2.5e-3, 1990.0
    t = [2000.0, 2005.0, 2010.0, 2015.0, 2020.0]
    d = [u * (ti - tr) for ti in t]
    res = phylog(t, d)
    assert abs(res["rate"] - u) < 1e-15
    assert abs(res["tmrca"] - tr) < 1e-9
    assert abs(res["correlation"] - 1.0) < 1e-12
    for r in res["residuals"]:
        assert abs(r) < 1e-15


def test_phylog_hand_ols():
    # t = (0, 1, 2), d = (1, 3, 4): tbar = 1, dbar = 8/3.
    # sxy = (-1)(1 - 8/3) + 0 + (1)(4 - 8/3) = 5/3 + 4/3 = 3,
    # sxx = 2 -> slope 1.5, intercept 8/3 - 1.5 = 7/6,
    # x-intercept = -(7/6)/1.5 = -7/9.
    res = phylog([0.0, 1.0, 2.0], [1.0, 3.0, 4.0])
    assert abs(res["rate"] - 1.5) < 1e-12
    assert abs(res["intercept"] - 7.0 / 6.0) < 1e-12
    assert abs(res["tmrca"] + 7.0 / 9.0) < 1e-12


def test_phylog_r2_matches_correlation_squared():
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    d = [0.1, 0.35, 0.5, 0.81, 0.95]
    res = phylog(t, d)
    assert abs(res["r_squared"] - res["correlation"] ** 2) < 1e-15
    assert phylogenetic_dating is phylog
