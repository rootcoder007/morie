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


def test_sharp_fuzzy_kink_match_rdrobust_nn():
    # rdrobust(y, x, h = 0.5) (vce = "nn"); fuzzy = fz; deriv = 1, p = 2
    import math

    from morie import rdd as R
    from morie.fn import _frame_core as pd
    from morie.fn.causrddc import causrddc

    n = 400
    x = [math.sin(1.37 * i) * 1.2 + 0.4 * math.cos(0.21 * i) for i in range(n)]
    tr = [int(v >= 0) for v in x]
    fz = [int((x[i] >= 0 and i % 5 != 0) or (x[i] < 0 and i % 7 == 0)) for i in range(n)]
    d = pd.DataFrame(
        {
            "x": x,
            "y": [1 + 0.8 * x[i] + 0.3 * x[i] ** 2 + 1.2 * tr[i] + 0.4 * math.sin(3.1 * i) for i in range(n)],
            "yf": [1 + 0.8 * x[i] + 2 * fz[i] + 0.4 * math.sin(3.1 * i) for i in range(n)],
            "yk": [1 + 0.8 * x[i] + 1.5 * max(x[i], 0) + 0.4 * math.sin(3.1 * i) for i in range(n)],
            "fz": fz,
        }
    )
    s = R.sharp_rdd(d, "y", "x", bandwidth=0.5)
    assert abs(s.estimate - 1.31895764244) <= 1e-10 and abs(s.std_error - 0.0843098682739) <= 1e-11
    f = R.fuzzy_rdd(d, "yf", "x", "fz", bandwidth=0.5)
    assert abs(f.estimate - 2.30166428120295) <= 1e-12 and abs(f.std_error - 0.275553805228321) <= 1e-12
    k = R.kink_rdd(d, "yk", "x", bandwidth=0.5)
    assert abs(k.estimate - 4.65174038836) <= 1e-9 and abs(k.std_error - 1.39440321497) <= 1e-10
    cc = causrddc(d["yf"].tolist(), x, treatment=fz, h=0.5, b=0.5)
    assert abs(cc["bias_corrected"] - 2.45321344528486) <= 1e-12 and abs(cc["se_robust"] - 0.40614527539738) <= 1e-12


def test_default_bandwidths_are_mserd_and_match_rdrobust():
    # rdbwselect(...) h, b and rdrobust(...) with default bandwidths
    import math

    from morie import rdd as R
    from morie.fn import _frame_core as pd
    from morie.fn.causrddc import rd_mserd_bandwidth

    n = 400
    x = [math.sin(1.37 * i) * 1.2 + 0.4 * math.cos(0.21 * i) for i in range(n)]
    tr = [int(v >= 0) for v in x]
    fz = [int((x[i] >= 0 and i % 5 != 0) or (x[i] < 0 and i % 7 == 0)) for i in range(n)]
    y = [1 + 0.8 * x[i] + 0.3 * x[i] ** 2 + 1.2 * tr[i] + 0.4 * math.sin(3.1 * i) for i in range(n)]
    yf = [1 + 0.8 * x[i] + 2 * fz[i] + 0.4 * math.sin(3.1 * i) for i in range(n)]
    yk = [1 + 0.8 * x[i] + 1.5 * max(x[i], 0) + 0.4 * math.sin(3.1 * i) for i in range(n)]
    d = pd.DataFrame({"x": x, "y": y, "yf": yf, "yk": yk, "fz": fz})
    for got, ref in (
        (rd_mserd_bandwidth(y, x), (0.584477848907144, 0.93283333886784)),
        (rd_mserd_bandwidth(yf, x, treatment=fz), (0.479323994236654, 0.824084209003945)),
        (rd_mserd_bandwidth(yk, x, p=2, deriv=1), (0.635424623301575, 0.965984594756331)),
    ):
        assert abs(got["h"] - ref[0]) <= 1e-9 * ref[0] and abs(got["b"] - ref[1]) <= 1e-9 * ref[1]
    for r, ref in (
        (R.sharp_rdd(d, "y", "x"), (1.29449761957, 0.0787346761035)),
        (R.fuzzy_rdd(d, "yf", "x", "fz"), (2.31807950054, 0.29104188413)),
        (R.kink_rdd(d, "yk", "x"), (4.41013383404, 0.999751323253)),
        (R.rdd_bias_corrected(d, "y", "x"), (1.31362793469, 0.0921089745395)),
    ):
        assert abs(r.estimate - ref[0]) <= 1e-8 * abs(ref[0]) and abs(r.std_error - ref[1]) <= 1e-8 * ref[1]


def test_density_test_matches_rddensity():
    # rddensity(x, c = 0, h = 0.5); rddensity(x); rddensity(x, p = 1); mass points
    import math

    from morie import rdd as R

    x = [math.sin(1.37 * i) * 1.2 + 0.4 * math.cos(0.21 * i) for i in range(400)]
    assert abs(R.cattaneo_density_test(x, bandwidth=0.5).statistic + 0.424998995462388) <= 1e-10
    r = R.cattaneo_density_test(x)
    assert abs(r.statistic + 0.328188049213095) <= 1e-9
    assert (
        abs(r.details["h_left"] - 0.566842258247533) <= 1e-9 and abs(r.details["h_right"] - 0.536752772079201) <= 1e-9
    )
    assert abs(R.cattaneo_density_test(x, p=1).statistic - 0.829783793810616) <= 1e-9
    xr = [round(v, 1) for v in x]
    assert abs(R.cattaneo_density_test(xr).statistic - 0.317121592199131) <= 1e-9
    assert abs(R.cattaneo_density_test(xr, bandwidth=0.6).statistic + 0.746116220795682) <= 1e-9
