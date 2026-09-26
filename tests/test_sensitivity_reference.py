"""Sensitivity analysis against EValue, sensemakr, konfound, tipr and closed forms."""


def test_sensitivity_matches_reference_packages():
    # references: EValue evalues.*, sensemakr, konfound::pkonfound, tipr::tip
    from morie import sensitivity as S

    def close(a, b, tol=1e-10):
        assert len(a) == len(b)
        for u, v in zip(a, b):
            assert abs(u - v) < tol

    o = S.e_value_or(2.5, 1.4, 4.0, prevalence=0.3)
    close([o.e_value_point, o.e_value_ci], [2.53971129469801, 1.6488166904897])
    o = S.e_value_hr(1.8, 1.2, 2.6, rare=True)
    close([o.e_value_point, o.e_value_ci], [3, 1.68989794855664])
    o = S.e_value_hr(1.8, 1.2, 2.6)
    close([o.e_value_point, o.e_value_ci], [2.36714327129528, 1.52553091757873])
    o = S.e_value_d(0.5, se=0.2)
    close([o.e_value_point, o.e_value_ci], [2.52914198186127, 1.44302956344269])
    ov = S.omitted_variable_bias(
        0.407404970966835,
        0.0578602283905763,
        196,
        0.2,
        0.2,
        benchmark_covariates={"x2": (6.9856611516024e-08, 0.265033083453015)},
        kd=2,
    )
    close([ov.rv_q, ov.rv_qa], [0.392125514801805, 0.302160713916727])
    b = ov.benchmark_bounds["x2"]
    close(
        [b["r2yz_dx"], b["adjusted_estimate"], b["adjusted_se"], b["adjusted_lower_ci"], b["adjusted_upper_ci"]],
        [0.721211021924157, 0.407147837764754, 0.0306287098615988, 0.346743696480369, 0.46755197904914],
    )
    ox = S.omitted_variable_bias(40.0, 1.0, 30, 0, 0)
    close([ox.rv_qa, ox.rv_q], [0.9789403653761, 0.981921804436961])
    k1 = S.konfound(0.4, 0.1, 150, 3)
    k2 = S.konfound(-0.12, 0.1, 150, 3)
    close(
        [k1["percent_bias_to_invalidate"], k1["rir"], k1["impact_threshold_confounder"], k1["beta_threshold"]],
        [50.5885109341772, 76, 0.182899388060936, 0.197645956263291],
    )
    close(
        [k2["percent_bias_to_invalidate"], k2["rir"], k2["impact_threshold_confounder"], k2["beta_threshold"]],
        [39.2853755934456, 59, 0.0540508223426875, -0.197645956263291],
    )
    close([S.tipping_point_smd(1.8, 0.5)["confounder_outcome_effect"]], [3.24], 1e-12)


def test_sensitivity_closed_forms():
    import math

    from morie import sensitivity as S
    from morie.fn import _stats_core as stats

    rb = S.rosenbaum_bounds(
        [1] * 30 + [0] * 12 + [1] * 5, [0] * 30 + [1] * 12 + [1] * 5, gamma_range=[1, 2], method="mcnemar"
    )
    # exact McNemar tail P(X >= 30 | n = 42): the binomial sf at 29
    assert abs(rb.p_upper[0] - stats.binom.sf(29, 42, 0.5)) < 1e-12
    assert abs(rb.p_upper[1] - stats.binom.sf(29, 42, 2 / 3)) < 1e-12
    rs = S.rosenbaum_bounds([1, 2, 3, 3, 5], [0, 0, 3, 3, 1], gamma_range=[1], method="sign")
    assert abs(rs.p_upper[0] - stats.binom.sf(2, 3, 0.5)) < 1e-12
    mb = S.manski_bounds([0.2, 0.9, 0.7], [0.1, 0.4], p_treated=0.6)
    assert abs(mb["width"] - 1) < 1e-12
    assert abs(mb["lower_bound"] - (0.6 * 0.6 - (0.25 * 0.4 + 0.6))) < 1e-12
    ba = S.bias_adjusted_estimate(0.5, 0.1, rr_ud=2, rr_eu=3)
    assert abs(ba["bias"] - math.log(2 * 3 / (2 + 3 - 1))) < 1e-12
    ba2 = S.bias_adjusted_estimate(0.5, 0.1, rr_ud=2, rr_eu=2, prevalence_confounder=0.2)
    assert abs(ba2["bias"] - math.log(1.4 / 1.2)) < 1e-12
    tp = S.tipping_point_analysis(0.5, 0.15, 100, 100)
    assert abs(tp.tipping_point - (0.5 - stats.norm.ppf(0.975) * 0.15)) < 1e-12
