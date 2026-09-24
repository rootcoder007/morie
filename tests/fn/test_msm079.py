"""Tests for msm079.mvsml_bayesian_regression_eq_6_11."""

from morie.fn import _array_core as np

from morie.fn.msm079 import mvsml_bayesian_regression_eq_6_11


def test_msm079_basic():
    """Test basic functionality of the BMTME full conditionals (eq. 6.11).

    Model: Y = 1_IJ mu' + X B + Z1 b1 + Z2 b2 + E, with
        b2 | Sigma_T, Sigma_E ~ MN(0, Sigma_E (x) G, Sigma_T).

    The function returns the two inverse-Wishart full conditionals of
    steps 5 and 6 (p.196). The "estimate" payload is scale_T[0][0],
    the (1,1) element of the post. scale matrix for Sigma_T.
    """
    rng = np.random.default_rng(42)

    # Minimal multi-trait configuration: I=1, J=1, t=1.
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

    # The function returns a RichResult (dict-like) with the documented keys.
    assert isinstance(result, dict)
    for key in ("estimate", "nu_T_post", "scale_T", "nu_E_post", "scale_E", "method"):
        assert key in result

    # The estimate is the (1,1) element of scale_T.
    assert float(result["estimate"]) == float(result["scale_T"][0][0])

    # Method label as documented.
    assert result["method"] == "BMTME full conditionals (MVSML 2022 eq. 6.11)"

    # Posterior scale matrices must be 2-D lists with one row and one column.
    sT = result["scale_T"]
    sE = result["scale_E"]
    assert isinstance(sT, list) and isinstance(sE, list)
    assert len(sT) == 1 and len(sE) == 1
    assert len(sT[0]) == 1 and len(sE[0]) == 1


def test_msm079_edge():
    """Test edge cases with minimum-size multi-trait inputs."""
    rng = np.random.default_rng(43)

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
    assert "scale_T" in result
    assert "scale_E" in result
