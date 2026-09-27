"""Core fixes checked against their R references: geepack, DescTools, mvtnorm, anova(lm)."""

import math

from morie.fn import _frame_core as pd
from morie.fn.anotwo import anova_twoway
from morie.fn.ccc import concordance_corr
from morie.fn.dnntt import dunnett_test
from morie.fn.gee import gee_regression
from morie.fn.prop_ci import proportion_ci

N = 20
X1 = [math.sin(1.1 * i) + 0.1 * i for i in range(N)]
X2 = [math.cos(0.7 * i) for i in range(N)]
Y = [1 + 0.5 * a - 0.3 * b + 0.2 * math.sin(3.3 * i) for i, (a, b) in enumerate(zip(X1, X2))]
CL = [i // 4 for i in range(N)]


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_gee_matches_geepack():
    # geeglm(y ~ x, id = cl, corstr = ..., control = geese.control(epsilon = 1e-13))
    ref = {
        "exchangeable": (
            [1.03801472401938, 0.436513717677338],
            [0.108147011877728, 0.0695202196566055],
            0.364548110491062,
            0.066144313685491,
        ),
        "ar1": (
            [1.025167284833, 0.449627311438093],
            [0.0992762141057171, 0.0684903081046328],
            0.517380259417559,
            0.0657527721389452,
        ),
        "independence": (
            [1.01276379523411, 0.460987084605944],
            [0.116174438259752, 0.0898132437156341],
            0.0,
            0.0656446526180047,
        ),
    }
    for cs, (b, se, a, g) in ref.items():
        r = gee_regression(Y, [[v] for v in X1], CL, corr_structure=cs, tol=1e-12, max_iter=200)
        assert all(close(u, v, 1e-8) for u, v in zip(r.coefficients.values(), b))
        assert all(close(u, v, 1e-7) for u, v in zip(r.se.values(), se))
        assert abs(r.extra["alpha"] - a) < 1e-7 and close(r.extra["scale"], g, 1e-8)
    cnt = [round(3 + 2 * abs(v)) for v in Y]
    r = gee_regression(cnt, [[v] for v in X1], CL, family="poisson", tol=1e-12, max_iter=200)
    assert all(close(u, v, 1e-9) for u, v in zip(r.coefficients.values(), [1.61423684477466, 0.146132417456451]))
    assert all(close(u, v, 1e-8) for u, v in zip(r.se.values(), [0.0502981825287144, 0.0297813134314384]))
    assert close(r.extra["alpha"], 0.338897742169293, 1e-8)


def test_ccc_matches_desctools():
    r = concordance_corr(X1, Y)
    # DescTools::CCC(x, y, ci = "z-transform") and ci = "asymptotic"
    assert close(r.estimate, 0.59824339226692769, 1e-14)
    assert close(r.ci_lower, 0.38254267739764947, 1e-13) and close(r.ci_upper, 0.75210290764044829, 1e-13)
    lo, hi = r.extra["asymptotic_ci"]
    assert close(lo, 0.41372033522548701, 1e-13) and close(hi, 0.78276644930836836, 1e-13)
    assert concordance_corr(list(range(10)), list(range(10))).ci_lower == 1.0


def test_dunnett_matches_pmvt():
    r = dunnett_test([5.1, 4.8, 5.5, 5.0, 4.9], [5.9, 6.1, 5.7, 6.3, 5.8], [5.2, 5.4, 4.9, 5.6, 5.3])
    p = [c["p_adj"] for c in r.extra["comparisons"]]
    # 1 - mvtnorm::pmvt(-|t|, |t|, corr = lambda lambda', df = 12), GenzBretz abseps = 1e-11
    assert close(p[0], 0.000244005001208913, 1e-9) and close(p[1], 0.328906070804265, 1e-9)
    r = dunnett_test([5.1, 4.8, 5.5, 5.0, 4.9, 5.2], [5.9, 6.1, 5.7, 6.3], [5.2, 5.4, 4.9, 5.6, 5.3, 5.0, 5.7])
    c = r.extra["comparisons"]
    assert close(c[0]["t"], 5.23979288689629, 1e-12) and close(c[1]["t"], 1.43695157250271, 1e-12)
    assert close(c[0]["p_adj"], 0.000241724598797788, 1e-9) and close(c[1]["p_adj"], 0.290411591498084, 1e-9)


def test_anova_twoway_unbalanced_matches_anova_lm():
    d = pd.DataFrame(
        {
            "y": [3.1, 4.2, 5.0, 3.6, 4.9, 5.8, 2.2, 2.9, 4.4, 2.7, 3.3, 4.1, 3.9, 2.6],
            "a": ["p"] * 6 + ["q"] * 8,
            "b": ["u", "v", "w"] * 4 + ["u", "w"],
        }
    )
    r = anova_twoway(d)
    # anova(lm(y ~ a + b))
    assert close(r.statistic, 8.94718115906096, 1e-12) and close(r.extra["f_b"], 3.90146001067526, 1e-12)
    assert close(r.extra["ss_resid"], 5.25311764705882, 1e-12)
    r = anova_twoway(d, interaction=True)
    # anova(lm(y ~ a * b))
    assert close(r.statistic, 9.04582426394778, 1e-12) and close(r.extra["f_ab"], 1.05512524175668, 1e-12)
    assert close(r.extra["p_ab"], 0.392023666436377, 1e-11)


def test_proportion_ci_wald_and_exact_alias():
    lo, hi = proportion_ci(7, 25, method="wald")
    half = 1.959963984540054 * math.sqrt(0.28 * 0.72 / 25)
    assert close(lo, 0.28 - half, 1e-13) and close(hi, 0.28 + half, 1e-13)
    assert proportion_ci(7, 25, method="exact") == proportion_ci(7, 25, method="clopper-pearson")


def test_gee_singular_working_correlation():
    import math

    import pytest

    x = [math.sin(i) for i in range(1, 21)]
    g = [i // 4 for i in range(20)]
    # identical residual pattern in every cluster: exchangeable alpha = -1/(m - 1), the singular boundary
    with pytest.raises(ValueError, match="working correlation is singular"):
        gee_regression([a + b for a, b in zip(x, [0.2, -0.1, 0.3, 0] * 5)], x, g)
    y = [a + b + 0.3 * math.cos(7 * i) for a, b, i in zip(x, [0.2, -0.1, 0.3, 0] * 5, range(1, 21))]
    assert abs(gee_regression(y, x, g).extra["alpha"] - -0.21085606432035597) < 1e-8
