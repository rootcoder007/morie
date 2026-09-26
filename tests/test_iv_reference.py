"""morie.iv against R ivreg/sandwich and linearmodels on fixed data."""

import math

from morie import iv as I
from morie.fn import _frame_core as pd


def _data():
    rows = []
    for i in range(80):
        z1 = math.sin(1.3 * i) + 0.3 * math.cos(0.7 * i)
        z2 = math.cos(2.1 * i + 0.5)
        w = math.sin(0.37 * i + 1.0)
        u = 0.6 * math.sin(3.7 * i + 0.2)
        v = 0.5 * u + 0.4 * math.cos(5.3 * i)
        d = 0.8 * z1 + 0.6 * z2 + 0.3 * w + v
        y = 1.0 + 1.5 * d + 0.7 * w + u + 0.2 * math.sin(7.1 * i) * z1
        rows.append((y, d, z1, z2, w, i % 8))
    return pd.DataFrame({k: [r[j] for r in rows] for j, k in enumerate(["y", "d", "z1", "z2", "w", "g"])})


def close(a, b, tol):
    return all(abs(x - y) <= tol * abs(y) for x, y in zip(a, b))


def test_tsls_robust_se_is_hc1_on_projected_regressors():
    # sqrt(diag(sandwich::vcovHC(ivreg(y ~ d + w | z1 + z2 + w), "HC1"))), order const, d, w
    r = I.tsls(_data(), "y", ["d"], ["z1", "z2"], ["w"], robust=True)
    assert close([float(v) for v in r.std_errors], [0.0505459824547257, 0.0667536171757581, 0.0725244634555991], 1e-10)


def test_gmm_and_cue_match_linearmodels():
    # IVGMM / IVGMMCUE(cov_type = "robust"): coefficients, J; CUE SEs
    g = I.gmm_iv(_data(), "y", ["d"], ["z1", "z2"], ["w"])
    assert close([float(v) for v in g.coefficients], [1.01651581791, 1.44250806159, 0.706747364412], 1e-10)
    # efficient form (G' S^-1 G)^-1 at the final residuals; linearmodels'
    # W-sandwich differs from it at O(1e-5) here
    assert close([float(v) for v in g.std_errors], [0.0488046582694, 0.065789704909, 0.0712173125091], 1e-4)
    j = I.hansen_j_test(_data(), "y", ["d"], ["z1", "z2"], ["w"])
    assert abs(j.statistic - 1.01179918825) <= 1e-9
    c = I.cue_gmm(_data(), "y", ["d"], ["z1", "z2"], ["w"])
    assert close([float(v) for v in c.coefficients], [1.01790404736, 1.43913637976, 0.708076314862], 1e-7)
    assert close([float(v) for v in c.std_errors], [0.0488667178074, 0.0659360392028, 0.0713319811849], 1e-7)


def test_endogeneity_and_weak_iv_tests_match_ivreg():
    d = _data()
    # summary(ivreg, diagnostics = TRUE): Wu-Hausman F; Durbin form Hausman
    assert abs(I.durbin_wu_hausman(d, "y", ["d"], ["z1", "z2"], ["w"]).statistic - 37.6708096) <= 1e-6
    assert abs(I.hausman_test(d, "y", ["d"], ["z1", "z2"], ["w"]).statistic - 25.5180054657815) <= 1e-9
    # anova(lm(y - b0 d ~ w), lm(y - b0 d ~ z1 + z2 + w)) F
    assert abs(I.anderson_rubin_test(d, "y", ["d"], ["z1", "z2"], ["w"], beta0=0.0).statistic - 58.040106940631) <= 1e-9
    assert (
        abs(I.anderson_rubin_test(d, "y", ["d"], ["z1", "z2"], ["w"], beta0=1.5).statistic - 0.765031056265945) <= 1e-9
    )


def test_iv_probit_second_stage_is_the_glm_probit():
    # glm(yb ~ d + w + v_hat, binomial(link = "probit"), epsilon = 1e-15)
    d = _data()
    d["yb"] = [int(v + 2 * math.sin(11.3 * i) > 1.8) for i, v in enumerate(d["y"].tolist())]
    r = I.iv_probit(d, "yb", ["d"], ["z1", "z2"], ["w"])
    ref_b = [-0.476581376436956, 0.618378748911756, 0.429780999807716, 0.553796580644084]
    ref_se = [0.16471496314994, 0.235595075775551, 0.228676603206474, 0.485157307680015]
    assert close([float(v) for v in r.coefficients], ref_b, 1e-7)
    assert close([float(v) for v in r.std_errors], ref_se, 1e-7)
