"""Tests for fzt41.fauzi_thm4_1_surv_bias_var."""

from morie.fn import _array_core as np

from morie.fn.fzt41 import fauzi_thm4_1_surv_bias_var


def test_fzt41_basic():
    """Test basic functionality against Theorem 4.1, Eqs. (4.10)-(4.16)."""
    t = 1.5
    n = 200
    h = 0.3
    surv = 0.65           # S_X(t)
    cdf = 0.35            # F_X(t)
    cumsurv = 0.80        # bar S_X(t)
    b1 = 0.42             # coefficient from (4.14)
    b2 = 0.17             # coefficient from (4.15)
    dg = 1.1              # g'(g^{-1}(t))
    density = 0.55        # f_X(t)
    mu2 = 1.0
    vw = 1.0 / np.sqrt(np.pi)

    result = fauzi_thm4_1_surv_bias_var(
        t, n, h, surv, cdf, cumsurv, b1, b2, dg, density, mu2, vw
    )

    # Returned object behaves dict-like; check required keys are present.
    keys = set(result.keys()) if hasattr(result, "keys") else set(result.__dict__)
    assert "biassurv" in keys
    assert "varsurv" in keys
    assert "biascum" in keys
    assert "varcum" in keys
    assert "vw" in keys
    assert "h" in keys
    assert "n" in keys
    assert "method" in keys

    # Reference values computed independently from the documented formula.
    expected_bias_s = -(h * h / 2.0) * b1 * mu2
    expected_var_s = surv * cdf / n - (h / n) * dg * density * vw
    expected_bias_c = (h * h / 2.0) * b2 * mu2
    expected_var_c = (2.0 * cumsurv - surv ** 2) / n

    assert result["biassurv"] == expected_bias_s
    assert result["varsurv"] == expected_var_s
    assert result["biascum"] == expected_bias_c
    assert result["varcum"] == expected_var_c
    assert result["h"] == float(h)
    assert result["n"] == int(n)
    assert result["vw"] == vw


def test_fzt41_edge():
    """Test with a different vw and that the Gaussian default is consistent."""
    t = 0.7
    n = 50
    h = 0.5
    surv = 0.4
    cdf = 0.6
    cumsurv = 0.3
    b1 = -0.25
    b2 = 0.8
    dg = 0.9
    density = 1.2

    # No vw provided -> default Gaussian kernel value 1/sqrt(pi).
    result = fauzi_thm4_1_surv_bias_var(
        t, n, h, surv, cdf, cumsurv, b1, b2, dg, density
    )
    expected_vw = 1.0 / np.sqrt(np.pi)
    assert result["vw"] == expected_vw

    expected_bias_s = -(h * h / 2.0) * b1 * 1.0
    expected_var_s = surv * cdf / n - (h / n) * dg * density * expected_vw
    expected_bias_c = (h * h / 2.0) * b2 * 1.0
    expected_var_c = (2.0 * cumsurv - surv ** 2) / n
    assert result["biassurv"] == expected_bias_s
    assert result["varsurv"] == expected_var_s
    assert result["biascum"] == expected_bias_c
    assert result["varcum"] == expected_var_c
