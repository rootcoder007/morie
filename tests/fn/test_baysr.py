"""Tests for baysr.bayes_r_prior."""

from morie.fn import _array_core as np

from morie.fn.baysr import bayes_r_prior


def _build_X(rng, n, p):
    """Build X as a list of lists so core.mat can iterate rows then floats."""
    rows = rng.normal(0, 1, (n, p))
    return [[float(rows[i, j]) for j in range(p)] for i in range(n)]


def test_baysr_basic():
    """Test basic functionality of bayes_r_prior against the documented formula."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    rng_pi = np.random.default_rng(7)
    n, p = 100, 5
    y = [float(v) for v in rng_y.normal(0, 1, n)]
    X = _build_X(rng_X, n, p)
    # Four Moser et al. classes: first must be exactly 0.
    sigma_classes = [0.0, 1e-4, 1e-3, 1e-2]
    # Dirichlet-friendly weights (positive), then the function normalises them.
    pi = [float(v) for v in rng_pi.normal(1.0, 0.1, 4)]
    result = bayes_r_prior(y, X, pi, sigma_classes)
    # Documented keys in the returned RichResult payload.
    assert "estimate" in result.payload
    assert "beta_samples" in result.payload
    assert "class_probs" in result.payload
    assert "pi" in result.payload
    assert "sigma_g2" in result.payload
    assert "sigma_e2" in result.payload
    assert "n_nonzero" in result.payload
    # Shapes match the documented argument meanings.
    assert len(result.payload["beta_samples"]) == p
    assert len(result.payload["class_probs"]) == p
    assert len(result.payload["class_probs"][0]) == len(sigma_classes)
    assert len(result.payload["pi"]) == len(sigma_classes)
    # estimate and sigma_g2 are the same scalar.
    assert result.payload["estimate"] == result.payload["sigma_g2"]
    # n_nonzero counts markers whose modal class is the spike (class 0),
    # so it must be between 0 and p inclusive.
    assert 0 <= result.payload["n_nonzero"] <= p
    # Independent recomputation of the documented return values from the
    # inputs and the fitted pieces.  We reconstruct the four reported
    # scalars purely from the formula rather than copying numbers back.
    sg2 = result.payload["sigma_g2"]
    se2 = result.payload["sigma_e2"]
    pi_hat = result.payload["pi"]
    gam = result.payload["class_probs"]
    beta = result.payload["beta_samples"]
    sc = result.payload["sigma_classes"]
    mu = sum(y) / float(len(y))
    # mu is the sample mean of y; check that the returned mu matches.
    assert result.payload["mu"] == mu
    # Mixture weights must sum to one.
    assert abs(sum(pi_hat) - 1.0) < 1e-9
    # Posterior class probabilities for each marker must sum to one.
    for row in gam:
        assert abs(sum(row) - 1.0) < 1e-9
    # Reconstruct the genetic-variance estimator independently.
    ss = 0.0
    wsum = 0.0
    for j in range(p):
        for k in range(1, len(sc)):
            if sc[k] > 0.0:
                ss += gam[j][k] * beta[j] * beta[j] / sc[k]
                wsum += gam[j][k]
    if wsum > 0.0:
        sg2_check = max(ss / wsum, 1e-12)
    else:
        sg2_check = sg2
    assert abs(sg2 - sg2_check) < 1e-12
    # Reconstruct the residual variance from the residuals r = y - mu - X beta.
    res = 0.0
    for i in range(n):
        pred = mu + sum(X[i][j] * beta[j] for j in range(p))
        diff = y[i] - pred
        res += diff * diff
    se2_check = max(res / n, 1e-12)
    assert abs(se2 - se2_check) < 1e-9
    # Reconstruct n_nonzero from class_probs: markers whose modal class is not 0.
    nz = 0
    for row in gam:
        mk = 0
        for k in range(len(row)):
            if row[k] > row[mk]:
                mk = k
        if mk != 0:
            nz += 1
    assert nz == result.payload["n_nonzero"]


def test_baysr_edge():
    """Edge case: defaults are valid and the function still runs."""
    rng_y = np.random.default_rng(43)
    rng_X = np.random.default_rng(42)
    n, p = 100, 5
    y = [float(v) for v in rng_y.normal(0, 1, n)]
    X = _build_X(rng_X, n, p)
    # No pi, no sigma_classes -> defaults (equal weights, Moser 4-class set).
    result = bayes_r_prior(y, X)
    assert "estimate" in result.payload
    assert "sigma_classes" in result.payload
    assert result.payload["sigma_classes"][0] == 0.0
    assert len(result.payload["sigma_classes"]) == 4
    assert len(result.payload["pi"]) == 4
