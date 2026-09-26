"""rdd_bias_corrected against rdrobust (Calonico, Cattaneo & Titiunik 2014).

The reference numbers are rdrobust's output for the deterministic design
below with h = 0.4, b = 0.6, p = 1, triangular kernel:
``rdrobust(y, x, c=0, p=1, h=0.4, b=0.6, kernel="triangular", vce=...)``,
columns Conventional coef, Bias-Corrected coef, Conventional se, Robust se.
"""

import math

import pytest

from morie import rdd
from morie.fn import _frame_core as pd

REF = {
    "nn": [1.1790443482497563, 1.17295736747749, 0.04424662248950165, 0.05462407381752308],
    "hc": [1.1790443482497563, 1.17295736747749, 0.07352579785169784, 0.08837320267727357],
}


def _design():
    n = 400
    x = [-1 + 2 * (i + 0.5) / n for i in range(n)]
    y = [0.4 + 0.8 * v - 0.5 * v * v + (1.2 if v >= 0 else 0) + 0.3 * math.sin(37.0 * i) for i, v in enumerate(x)]
    return pd.DataFrame({"y": y, "x": x})


@pytest.mark.parametrize("vce", ["nn", "hc"])
def test_rdd_bias_corrected_matches_rdrobust(vce):
    tau_c, tau_bc, se_c, se_rb = REF[vce]
    r = rdd.rdd_bias_corrected(_design(), "y", "x", 0.0, bandwidth=0.4, rho=0.4 / 0.6, vce=vce)
    assert abs(r.estimate - tau_bc) <= 1e-9 * abs(tau_bc)
    assert abs(r.std_error - se_rb) <= 1e-9 * se_rb
    assert abs(r.details["tau_conventional"] - tau_c) <= 1e-9 * abs(tau_c)
    assert abs(r.details["se_conventional"] - se_c) <= 1e-9 * se_c
    z = 1.959963984540054
    assert abs(r.ci_lower - (tau_bc - z * se_rb)) <= 1e-9
    assert abs(r.ci_upper - (tau_bc + z * se_rb)) <= 1e-9


def test_rdd_bias_corrected_rejects_non_compact_kernel():
    with pytest.raises(ValueError, match="kernel must be one of"):
        rdd.rdd_bias_corrected(_design(), "y", "x", 0.0, bandwidth=0.4, kernel="gaussian")
