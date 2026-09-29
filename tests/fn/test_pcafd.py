"""Tests for pca_features."""

import pytest

from morie.fn import _array_core as np
from morie.fn.pcafd import pca_features, pcafd


def test_basic():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    r = pca_features(X, n_components=2)
    assert r.extra["n_components"] == 2
    assert len(r.extra["explained_variance_ratio"]) == 2


def test_alias():
    assert pcafd is pca_features


def test_too_few():
    with pytest.raises(ValueError):
        pca_features([[1, 2]])


def test_explained_variance_ratio_recomputed():
    import math

    X = [[1.0, 2.0], [2.0, 3.5], [3.0, 3.0], [4.0, 6.0], [5.0, 5.5]]
    n = 5
    m = [sum(r[j] for r in X) / n for j in range(2)]
    s = [[sum((r[a] - m[a]) * (r[b] - m[b]) for r in X) / (n - 1) for b in range(2)] for a in range(2)]
    tr, det = s[0][0] + s[1][1], s[0][0] * s[1][1] - s[0][1] ** 2
    d = math.sqrt(tr * tr / 4 - det)
    ev = [tr / 2 + d, tr / 2 - d]
    r = pca_features(X, n_components=1)
    assert r.extra["eigenvalues"] == pytest.approx(ev, rel=1e-10)
    assert r.value == pytest.approx(ev[0] / sum(ev), rel=1e-10)
