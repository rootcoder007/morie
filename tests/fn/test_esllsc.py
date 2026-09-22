"""Tests for esllsc.esl_lda_disc."""

from morie.fn import _array_core as np

from morie.fn.esllsc import esl_lda_disc


def test_esllsc_basic():
    """Test basic functionality."""
    rng_x = np.random.default_rng(42)
    rng_y = np.random.default_rng(43)
    # Two well-separated classes in 5-D so n > K and the pooled covariance
    # is well defined.
    X = np.vstack([
        rng_x.normal(loc=-2.0, scale=1.0, size=(50, 5)),
        rng_x.normal(loc=+2.0, scale=1.0, size=(50, 5)),
    ])
    y = np.array([0] * 50 + [1] * 50)
    result = esl_lda_disc(X, y)
    assert isinstance(result, dict)
    # Required payload keys per the docstring.
    for key in (
        "estimate",
        "prediction",
        "discriminants",
        "classes",
        "priors",
        "means",
        "pooled_covariance",
        "n",
        "p",
        "K",
        "method",
    ):
        assert key in result
    assert result["n"] == 100
    assert result["p"] == 5
    assert result["K"] == 2
    assert sorted(result["classes"]) == [0, 1]
    assert result["priors"] == [0.5, 0.5]
    assert len(result["prediction"]) == 100
    assert result["estimate"] in (0, 1)
    # Independent check of the discriminant for the first training point.
    classes = sorted(set(y.tolist()), key=repr)
    n, p = X.shape
    K = len(classes)
    means = []
    priors = []
    S = np.zeros((p, p))
    for c in classes:
        Xi = X[y == c]
        mu = Xi.mean(axis=0)
        means.append(mu)
        priors.append(Xi.shape[0] / n)
        C = Xi - mu
        S += C.T @ C
    S = S / (n - K)
    Sinv = np.linalg.inv(S)
    x0 = X[0]
    D0 = [float(x0 @ Sinv @ mu - 0.5 * float(mu @ Sinv @ mu) + np.log(pi_))
          for m, pi_ in zip(means, priors)]
    expected_first_pred = classes[int(np.argmax(np.array(D0)))]
    assert result["prediction"][0] == expected_first_pred


def test_esllsc_edge():
    """Test edge cases: querying a new point."""
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=-3.0, scale=1.0, size=(40, 3)),
        rng.normal(loc=+3.0, scale=1.0, size=(40, 3)),
    ])
    y = np.array([0] * 40 + [1] * 40)
    # A query point sits between the two class means.
    query = [[0.0, 0.0, 0.0]]
    result = esl_lda_disc(X, y, query=query)
    assert isinstance(result, dict)
    assert len(result["prediction"]) == 1
    assert result["estimate"] in (0, 1)
    # Discriminants are row-major n_query x K.
    assert len(result["discriminants"]) == 1 * 2
    # Independent computation of the discriminant at the query point.
    classes = sorted(set(y.tolist()), key=repr)
    n, p = X.shape
    K = len(classes)
    means = []
    priors = []
    S = np.zeros((p, p))
    for c in classes:
        Xi = X[y == c]
        mu = Xi.mean(axis=0)
        means.append(mu)
        priors.append(Xi.shape[0] / n)
        C = Xi - mu
        S += C.T @ C
    S = S / (n - K)
    Sinv = np.linalg.inv(S)
    q = np.asarray(query, dtype=float).reshape(-1, p)[0]
    expected_disc = [float(q @ Sinv @ mu - 0.5 * float(mu @ Sinv @ mu) + np.log(pi_))
                     for mu, pi_ in zip(means, priors)]
    expected_pred = classes[int(np.argmax(np.array(expected_disc)))]
    assert result["prediction"][0] == expected_pred
