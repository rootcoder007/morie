"""Tests for bmtme_conditionals.bmtme_conditionals."""

from morie.fn import _array_core as np

from morie.fn.bmtme_conditionals import bmtme_conditionals


def test_msm076_basic():
    """Test basic functionality of the BMTME full conditionals (eq. 6.11).

    Model: Y = 1_IJ mu' + X B + Z1 b1 + Z2 b2 + E, with
        b2 | Sigma_T, Sigma_E ~ MN(0, Sigma_E (x) G, Sigma_T).

    The function returns the two inverse-Wishart full conditionals of
    steps 5 and 6 (p.196). The "estimate" payload is scale_T[0][0],
    the (1,1) element of the post. scale matrix for Sigma_T.
    """
    rng = np.random.default_rng(42)

    # One genotype (I=1), one environment (J=1) -> a single observation row.
    # 2-D matrices as required by the documented multi-trait shapes.
    Y = rng.normal(0.0, 1.0, (1, 1))
    Z1 = rng.normal(0.0, 1.0, (1, 1))   # incidence for b1 (genotype main effect)
    Z2 = rng.normal(0.0, 1.0, (1, 1))   # incidence for b2 (G x E interaction)

    # Covariance matrices: square, symmetric positive definite (documented constraint).
    G = np.eye(1)                         # genomic relationship (1 genotype)
    Sigma_T = np.eye(1)                   # residual covariance across traits (1 trait)
    Sigma_E = np.eye(1)                   # covariance across environments (1 env)
    R = np.eye(1)                         # residual variance (1 trait)

    # b1 and b2 must be 2-D (matrix-normal layout): columns index traits.
    b1 = rng.normal(0.0, 1.0, (1, 1))
    b2 = rng.normal(0.0, 1.0, (1, 1))

    result = bmtme_conditionals(Y, Z1, Z2, G, Sigma_T, Sigma_E, R,
                                b1=b1, b2=b2)

    # Documented return keys (from the implementation).
    assert isinstance(result, dict)
    assert "estimate" in result
    for key in ("nu_T_post", "scale_T", "nu_E_post", "scale_E", "method"):
        assert key in result

    # `estimate` is the (1,1) element of scale_T, computed independently.
    assert float(result["estimate"]) == float(result["scale_T"][0][0])

    # Method label as documented.
    assert result["method"] == "BMTME full conditionals (MVSML 2022 eq. 6.11)"

    # Posterior degrees of freedom and scale matrices must be 2-D.
    sT = np.asarray(result["scale_T"])
    sE = np.asarray(result["scale_E"])
    assert sT.ndim == 2 and sE.ndim == 2
    assert sT.shape == (1, 1)
    assert sE.shape == (1, 1)


def test_msm076_edge():
    """Test edge cases: minimum-size multi-trait inputs (I=J=t=1)."""
    rng = np.random.default_rng(42)

    Y = rng.normal(0.0, 1.0, (1, 1))
    Z1 = rng.normal(0.0, 1.0, (1, 1))
    Z2 = rng.normal(0.0, 1.0, (1, 1))
    G = np.eye(1)
    Sigma_T = np.eye(1)
    Sigma_E = np.eye(1)
    R = np.eye(1)
    b1 = rng.normal(0.0, 1.0, (1, 1))
    b2 = rng.normal(0.0, 1.0, (1, 1))

    result = bmtme_conditionals(Y, Z1, Z2, G, Sigma_T, Sigma_E, R,
                                b1=b1, b2=b2)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "scale_T" in result
    assert "scale_E" in result
