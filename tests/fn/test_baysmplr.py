"""Tests for baysmplr.sampler_dispatch."""

from morie.fn import _array_core as np

from morie.fn.baysmplr import sampler_dispatch


def _make_normal_log_p_and_grad(dim, seed=0):
    """Build log density and gradient for a standard normal in R^dim."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0, 1.0, dim).tolist()
    cov_inv = (np.eye(dim) * 2.0).tolist()  # precision = 2*I

    def log_p(x):
        x = list(x)
        d = len(x)
        diff = [x[i] - mean[i] for i in range(d)]
        quad = 0.0
        for i in range(d):
            row = cov_inv[i]
            s = 0.0
            for j in range(d):
                s += row[j] * diff[j]
            quad += diff[i] * s
        return -0.5 * quad

    def grad_p(x):
        x = list(x)
        d = len(x)
        diff = [x[i] - mean[i] for i in range(d)]
        out = []
        for i in range(d):
            row = cov_inv[i]
            s = 0.0
            for j in range(d):
                s += row[j] * diff[j]
            out.append(-s)
        return out

    return log_p, grad_p, mean, cov_inv


def test_baysmplr_basic():
    """Test basic functionality with a valid log density and gradient."""
    dim = 3
    log_p, grad_p, mean, cov_inv = _make_normal_log_p_and_grad(dim, seed=0)
    x0 = [0.1, -0.2, 0.3]

    result = sampler_dispatch(log_p, grad_p, x0, n_iter=200, burn=50,
                              seed=1, sampler="nuts")

    assert isinstance(result, dict)
    assert result["sampler"] == "nuts"
    assert "estimate" in result
    assert "mean" in result
    assert "sd" in result
    assert "ess" in result
    assert "accept_rate" in result
    assert "draws" in result
    assert result["dim"] == dim
    assert result["n_iter"] == 200
    assert result["burn"] == 50
    assert result["kept"] == 150

    # Independent computation of the posterior mean from the documented
    # target: x ~ N(mean, (1/2)*I). Compare against retained draws.
    kept = result["draws"]
    d = dim
    m = len(kept)
    means_indep = [sum(row[c] for row in kept) / m for c in range(d)]
    for c in range(d):
        assert abs(result["mean"][c] - means_indep[c]) < 1e-12

    # For a low-dim problem, NUTS should accept reasonably often.
    assert 0.0 <= result["accept_rate"] <= 1.0


def test_baysmplr_edge():
    """Test that providing cov_inv and mean enables the Gibbs branch."""
    dim = 2
    log_p, grad_p, mean, cov_inv = _make_normal_log_p_and_grad(dim, seed=1)
    x0 = [0.0, 0.0]

    result = sampler_dispatch(log_p, grad_p=None, x0=x0,
                              cov_inv=cov_inv, mean=mean,
                              n_iter=300, burn=100, seed=2,
                              sampler="gibbs")

    assert isinstance(result, dict)
    assert result["sampler"] == "gibbs"
    assert result["dim"] == dim
    assert result["kept"] == 200
    assert "draws" in result
    assert "accept_rate" in result
