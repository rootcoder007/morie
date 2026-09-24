"""Tests for msm080.mvsml_bayesian_regression_eq_6_11."""

from morie.fn import _array_core as np

from morie.fn.msm080 import mvsml_bayesian_regression_eq_6_11


def test_msm080_basic():
    """Test basic functionality.

    Model: Y = 1_IJ mu' + X B + Z1 b1 + Z2 b2 + E, with
        b2 | Sigma_T, Sigma_E ~ MN(0, Sigma_E (x) G, Sigma_T).

    The function returns the two inverse-Wishart full conditionals of
    steps 5 and 6 (p.196). The "estimate" payload is scale_T[0][0].
    """
    rng = np.random.default_rng(42)

    # Multi-trait matrices (2-D) as documented by bmtme_conditionals.
    Y = rng.normal(0.0, 1.0, (1, 1))
    Z1 = rng.normal(0.0, 1.0, (1, 1))
    Z2 = rng.normal(0.0, 1.0, (1, 1))
    G = np.eye(1)
    Sigma_T = np.eye(1)
    Sigma_E = np.eye(1)
    R = np.eye(1)
    b1 = rng.normal(0.0, 1.0, (1, 1))
    b2 = rng.normal(0.0, 1.0, (1, 1))

    result = mvsml_bayesian_regression_eq_6_11(Y, Z1, Z2, G, Sigma_T, Sigma_E, R,
                                               b1=b1, b2=b2)

    assert isinstance(result, dict)
    assert "estimate" in result
    for key in ("nu_T_post", "scale_T", "nu_E_post", "scale_E", "method"):
        assert key in result

    # `estimate` is the (1,1) element of scale_T.
    assert float(result["estimate"]) == float(result["scale_T"][0][0])

    # Method label as documented.
    assert result["method"] == "BMTME full conditionals (MVSML 2022 eq. 6.11)"

    # Posterior scale matrices must be 2-D with matching shape.
    sT = np.asarray(result["scale_T"])
    sE = np.asarray(result["scale_E"])
    assert sT.ndim == 2 and sE.ndim == 2
    assert sT.shape == (1, 1)
    assert sE.shape == (1, 1)


def test_msm080_edge():
    """Test edge cases: minimum-size multi-trait inputs (I=J=t=1).

    Omitting the optional b1 / b2 keyword arguments exercises the
    documented default of b1=b2=None.
    """
    rng = np.random.default_rng(42)

    Y = rng.normal(0.0, 1.0, (1, 1))
    Z1 = rng.normal(0.0, 1.0, (1, 1))
    Z2 = rng.normal(0.0, 1.0, (1, 1))
    G = np.eye(1)
    Sigma_T = np.eye(1)
    Sigma_E = np.eye(1)
    R = np.eye(1)

    # Call without optional b1 / b2; the documented signature allows None.
    result = mvsml_bayesian_regression_eq_6_11(Y, Z1, Z2, G, Sigma_T, Sigma_E, R)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scale_T" in result
    assert "scale_E" in result
    assert "method" in result

    sT = np.asarray(result["scale_T"])
    sE = np.asarray(result["scale_E"])
    assert sT.ndim == 2 and sE.ndim == 2
    assert sT.shape == (1, 1)
    assert sE.shape == (1, 1)
