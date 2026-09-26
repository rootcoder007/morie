"""morie.multiple_testing against qvalue and p.adjust."""


def test_estimate_pi0_follows_qvalue():
    from morie import multiple_testing as M

    p = [(k - 0.5) / 60 for k in range(1, 61)]
    p = sorted(p[0:60:2] + [v**3 for v in p[:30]])
    # qvalue::pi0est(p, pi0.method = "bootstrap") exactly; the smoother is
    # the exact df = 3 natural spline, which smooth.spline matches to its
    # own df tolerance
    assert abs(M.estimate_pi0(p, "bootstrap") - 0.49019607843137253) <= 1e-14
    assert abs(M.estimate_pi0(p, "storey") - 0.4275865919952116) <= 1e-5
    # BKY stage one at q / (1 + q)
    r = M.benjamini_hochberg(p, alpha=0.05 / 1.05).n_rejected
    assert M.estimate_pi0(p, "two_step") == (60 - r) / 60
