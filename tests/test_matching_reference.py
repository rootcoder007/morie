"""morie.matching balance and propensity against cobalt and glm."""

import math

from morie import matching as M
from morie.fn import _frame_core as pd


def _data():
    n = 60
    x1 = [round(math.sin(1.1 * i) * 2 + 0.3 * i / 10, 3) for i in range(n)]
    x2 = [round(math.cos(0.7 * i) + 0.5 * math.sin(2.9 * i), 3) for i in range(n)]
    t = [int(math.sin(1.9 * i + 0.3) + 0.5 * x1[i] > 0.2) for i in range(n)]
    w = [round(1 + 0.5 * abs(math.sin(0.9 * i)) + 0.3 * t[i], 3) for i in range(n)]
    return pd.DataFrame({"t": t, "x1": x1, "x2": x2, "w": w})


def close(a, b, tol):
    return all(abs(x - y) <= tol * abs(y) for x, y in zip(a, b))


def test_balance_matches_cobalt():
    # cobalt::bal.tab(t ~ x1 + x2, weights = "w", s.d.denom = "pooled", stats = c("m", "v", "ks"))
    r = M.balance_diagnostics(_data(), "t", ["x1", "x2"], weights="w").balance_table
    assert close(r["smd"].tolist(), [1.67876185466111, 0.274662902480088], 1e-12)
    assert close(r["variance_ratio"].tolist(), [1.48872009615744, 1.06429541438012], 1e-12)
    assert close(r["ks_stat"].tolist(), [0.638985722883579, 0.202341881639128], 1e-12)
    u = M.balance_diagnostics(_data(), "t", ["x1", "x2"]).balance_table
    assert close(u["smd"].tolist(), [1.70640717405192, 0.333769817807865], 1e-12)


def test_propensity_is_the_glm_logit():
    # fitted(glm(t ~ x1 + x2, family = binomial))
    ps = M.estimate_propensity_score(_data(), "t", ["x1", "x2"]).tolist()
    assert close([ps[0], ps[1], ps[59]], [0.582605097830043, 0.936129507189641, 0.98336918715535], 1e-9)


def test_abadie_imbens_se_matches_matching_package():
    # Match(y, t, X, estimand = "ATT", M = 1, sample = TRUE, Weight = 1, Var.calc = 1)$se
    n = 60
    x1 = [math.sin(1.1 * i) * 2 + 0.3 * i / 10 for i in range(n)]
    x2 = [math.cos(0.7 * i) + 0.5 * math.sin(2.9 * i) for i in range(n)]
    t = [int(math.sin(1.9 * i + 0.3) + 0.5 * x1[i] > 0.2) for i in range(n)]
    y = [1 + 2 * t[i] + x1[i] + 0.5 * x2[i] + 0.7 * math.sin(4.3 * i) for i in range(n)]
    d = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})

    def var(v):
        m = sum(v) / n
        return sum((a - m) ** 2 for a in v) / (n - 1)

    s1, s2 = 1 / var(x1), 1 / var(x2)
    tr = [i for i in range(n) if t[i] == 1]
    co = [i for i in range(n) if t[i] == 0]
    ctrl = [min(co, key=lambda j: s1 * (x1[j] - x1[k]) ** 2 + s2 * (x2[j] - x2[k]) ** 2) for k in tr]
    p = pd.DataFrame({"treated_idx": tr, "control_idx": ctrl})
    se = M.abadie_imbens_se(d, "y", "t", p, covariates=["x1", "x2"])
    assert abs(se - 0.256974848359553) <= 1e-12
