"""morie.causal propensity and AIPW against glm/lm on fixed data."""

import math

from morie import causal as C
from morie.fn import _frame_core as pd


def _data():
    n = 40
    x = [round(math.sin(1.7 * i) * 1.3, 3) for i in range(n)]
    g = [("north", "south", "west")[(i * 7) % 3] for i in range(n)]
    t = [int(math.sin(2.3 * i + 0.4) + 0.6 * x[i] + 0.5 * (g[i] == "west") > 0.1) for i in range(n)]
    y = [round(1 + 0.5 * t[i] + x[i] + (g[i] == "west") + 0.4 * math.cos(3.1 * i), 3) for i in range(n)]
    return pd.DataFrame({"t": t, "y": y, "x": x, "g": g})


def test_propensity_is_the_glm_logit_with_dummies():
    # fitted(glm(t ~ x + g, family = binomial)): g enters as dummies
    ps = C.compute_propensity_scores(_data(), "t", ["x", "g"]).tolist()
    for i, ref in ((0, 0.77712827762826375), (1, 0.89709216378994716), (39, 0.69359812529157316)):
        assert abs(ps[i] - ref) <= 1e-10


def test_aipw_matches_glm_lm_influence_function():
    # separate lm(y ~ x + g) per arm, propensity clipped to [0.01, 0.99]
    a = C.estimate_aipw(_data(), treatment="t", outcome="y", covariates=["x", "g"], outcome_model="linear")
    assert abs(a["ate"] - 0.5726329578661663) <= 1e-10
    assert abs(a["se"] - 0.091279793606902818) <= 1e-10
