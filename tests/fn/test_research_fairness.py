import math

import pytest

from morie.fn.research_fairness import (
    fairness_base_rate_bounds,
    fairness_compare_groups,
    fairness_implied_fpr,
    fairness_rates,
    fairness_true_rate,
    hazard_selection,
    logit_rescale,
    ranking_resolution,
)


def test_chouldechova_identity_reproduces_the_table_fpr():
    r = fairness_rates(61, 37.5, 22, 179)
    assert r.n == 299.5
    assert r.p == pytest.approx(83 / 299.5, abs=1e-15)
    assert fairness_implied_fpr(r.p, r.ppv, r.fnr) == pytest.approx(r.fpr, abs=1e-12)
    # equal ppv and fnr with different base rates force different fpr
    assert fairness_implied_fpr(0.3, 0.6, 0.2) != fairness_implied_fpr(0.4, 0.6, 0.2)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        fairness_implied_fpr(1.0, 0.5, 0.1)


def test_base_rate_bounds_attained_by_the_box_corners():
    a, b = 0.03, 0.21
    ps = [0.07, 0.31, 0.95]
    bb = fairness_base_rate_bounds(ps, a, b)
    for k, p in enumerate(ps):
        assert bb["lower"][k] == pytest.approx(max(0.0, fairness_true_rate(p, a, 0.0)), abs=1e-15)
        assert bb["upper"][k] == pytest.approx(min(1.0, fairness_true_rate(p, 0.0, b)), abs=1e-15)
        for al in (0.0, a):
            for be in (0.0, b):
                t = fairness_true_rate(p, al, be)
                if 0 <= t <= 1:
                    assert bb["lower"][k] - 1e-12 <= t <= bb["upper"][k] + 1e-12
    assert fairness_true_rate(ps, a, b) == pytest.approx([(p - a) / (1 - a - b) for p in ps], abs=1e-15)


def test_compare_groups_order_and_breakdown():
    c = fairness_compare_groups(0.11, 0.29, 0.02, 0.2)
    assert c.decided and c.order == "a < b"
    lo_b = (0.29 - 0.02) / 0.98
    assert c.breakdown_beta_max == pytest.approx(1 - 0.11 / lo_b, abs=1e-15)
    # at the breakdown box the upper end of a meets the lower end of b
    assert 0.11 / (1 - c.breakdown_beta_max) == pytest.approx(lo_b, abs=1e-12)
    assert fairness_compare_groups(0.37, 0.13, 0.04, 0.1).order == "b < a"
    u = fairness_compare_groups(0.2, 0.22, 0.02, 0.2)
    assert u.order == "undecided" and u.theorem == "Research.P5.compare_undecided"


def test_logit_rescale():
    r = logit_rescale([0.8, -1.3, 2.1], [1.0, 0.25, 3.0])
    s2 = math.pi**2 / 3
    for k, (b, v) in enumerate(zip([0.8, -1.3, 2.1], [1.0, 0.25, 3.0])):
        c = math.sqrt(s2 / (s2 + v))
        assert r.rescale[k] == pytest.approx(c, abs=1e-15)
        assert r.apparent_change[k] == pytest.approx(math.exp(b * c) / math.exp(b), rel=1e-12)
    p = logit_rescale(0.7, 0.0, error_var=1)
    assert p.rescale == 1.0 and p.apparent_change == 1.0


def test_ranking_resolution():
    est = [0.21, 0.35, 0.8, 0.52, 0.5]
    hw = [0.1, 0.07, 0.05, 0.02, 0.2]
    r = ranking_resolution(est, hw)
    pairs = [(i, j) for j in range(5) for i in range(j)]
    ident = {(i, j): abs(est[i] - est[j]) > hw[i] + hw[j] for i, j in pairs}
    for (i, j), v in ident.items():
        assert r.identified_pairs[i][j] is v and r.identified_pairs[j][i] is v
    assert r.n_pairs == 10
    assert r.share_unidentified == pytest.approx(sum(not v for v in ident.values()) / 10, abs=1e-15)
    assert r.resolution == 0.4


def test_hazard_selection_from_depletion_alone():
    s, h, lo = 0.37, 0.61, 0.13
    r = hazard_selection(s, h, lo, (0.43, 0.77), (0.88, 0.91))
    for arm, sh, sl in (("control", 0.43, 0.88), ("treated", 0.77, 0.91)):
        w = s * sh / (s * sh + (1 - s) * sl)
        assert r.surviving_high_share[arm] == pytest.approx(w, abs=1e-15)
        assert r.period2_hazard[arm] == pytest.approx(w * h + (1 - w) * lo, abs=1e-15)
    assert r.period2_hazard_ratio > 1  # treated depletes the high-risk type less
    with pytest.raises(ValueError, match="l < h"):
        hazard_selection(0.5, 0.1, 0.5, (1, 1), (1, 1))
