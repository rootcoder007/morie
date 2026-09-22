"""Tests for aitlnp.logistic_normal_pdf."""

from morie.fn import _array_core as np

from morie.fn.aitlnp import logistic_normal_pdf


def test_aitlnp_basic():
    """Test basic functionality."""
    x = np.array([0.3, 0.3, 0.4])
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = logistic_normal_pdf(x, mu, Sigma)
    assert isinstance(result, dict)
    assert "density" in result
    assert "log_density" in result
    assert "alr" in result
    assert "quadratic_form" in result
    assert "log_jacobian" in result
    assert "log_det" in result
    assert "D" in result

    # Verify density equals exp(log_density).
    assert abs(result["density"] - np.exp(result["log_density"])) < 1e-12

    # Verify log_jacobian = -sum(log(x_i)) over ALL D parts.
    lj = -sum(np.log(x[i]) for i in range(len(x)))
    assert abs(result["log_jacobian"] - lj) < 1e-12

    # Verify alr: log(x_i) - log(x_D) for i=1..D-1.
    D = len(x)
    expected_alr = [np.log(x[i]) - np.log(x[D - 1]) for i in range(D - 1)]
    for a, b in zip(result["alr"], expected_alr):
        assert abs(a - b) < 1e-12

    # Verify quadratic form: (alr - mu)' Sigma^{-1} (alr - mu).
    dv = [expected_alr[i] - mu[i] for i in range(D - 1)]
    # Solve Sigma * z = dv -> z = Sigma^{-1} dv.
    z = np.linalg.solve(Sigma, dv)
    expected_q = sum(dv[i] * z[i] for i in range(D - 1))
    assert abs(result["quadratic_form"] - expected_q) < 1e-12

    # Verify log_det = log|Sigma| via SciPy reference.
    sign, logdet = np.linalg.slogdet(Sigma)
    assert abs(result["log_det"] - logdet) < 1e-10
    assert sign > 0

    # Independent formula computation for this D=3, identity Sigma, mu=0 case:
    # f(x) = (2 pi)^{-(D-1)/2} * |Sigma|^{-1/2} * (prod x_i)^{-1}
    #        * exp(-1/2 * (alr - mu)' Sigma^{-1} (alr - mu))
    D = len(x)
    prod_x = 1.0
    for v in x:
        prod_x *= v
    log_norm = (-0.5 * (D - 1) * np.log(2.0 * np.pi)
                - 0.5 * logdet
                - np.log(prod_x)
                - 0.5 * expected_q)
    expected_density = np.exp(log_norm)
    assert abs(result["density"] - expected_density) < 1e-10
    assert abs(result["log_density"] - log_norm) < 1e-10

    # Density must be strictly positive.
    assert result["density"] > 0.0

    # D is recorded.
    assert result["D"] == float(D)


def test_aitlnp_edge():
    """Test edge cases."""
    x = np.array([0.5, 0.5])
    mu = np.array([0.0])
    Sigma = np.array([[1.0]])
    result = logistic_normal_pdf(x, mu, Sigma)
    assert isinstance(result, dict)
    assert "density" in result
    assert result["D"] == 2.0
    assert result["density"] > 0.0
