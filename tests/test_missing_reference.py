"""morie.missing against mice and norm on fixed data."""


def test_rubins_rules_and_little_match_mice_and_norm():
    # mice::pool.scalar; norm::em.norm(criterion = 1e-14) plugged into Little (1988, eq. 2)
    from morie import missing as M
    from morie.fn import _frame_core as pd

    Q = [1.2, 1.5, 1.1, 1.4, 1.3]
    U = [0.04, 0.05, 0.045, 0.05, 0.042]
    r = M.rubins_rules(Q, U)
    assert abs(r.total_variance - 0.075399999999999981) <= 1e-15
    assert abs(r.df - 25.267377777777789) <= 1e-11
    assert abs(r.fmi - 0.44047988516312442) <= 1e-13
    assert abs(r.relative_efficiency - 1 / (1 + r.fmi / 5)) <= 1e-15
    r2 = M.rubins_rules(Q, U, dfcom=28)
    assert abs(r2.df - 9.7104933353729557) <= 1e-11
    assert abs(r2.fmi - 0.49262206679320941) <= 1e-13
    nan = float("nan")
    a = [5.1, 6.3, nan, 7.2, 5.9, 6.6, 5.4, nan, 5.5, 6.8, 4.9, 6.1, 5.7, nan, 6.4, 5.2, 6.9, 5.8, 6.2, 5.6]
    b = [4.2, nan, 3.9, 5.8, 4.4, nan, 5.3, 4.1, 4.6, 5.2, 3.8, 4.9, nan, 4.5, 5.1, 4.0, 5.6, 4.7, nan, 4.3]
    c = [6.1, 5.2, 6.9, nan, 6.0, 5.8, 7.1, 6.6, nan, 6.4, 5.5, 6.3, 6.0, 6.7, nan, 5.4, 7.0, 6.2, 6.5, 5.9]
    lt = M.littles_mcar_test(pd.DataFrame({"a": a, "b": b, "c": c}))
    assert abs(lt.test_statistic - 17.308859080868178) <= 1e-9 * 17.3
    assert lt.df == 6
    assert abs(lt.p_value - 0.0082127166597023536) <= 1e-8 * 0.0082
