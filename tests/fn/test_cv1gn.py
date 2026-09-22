"""Tests for cv1gn.cv1_genomic."""

import math

from morie.fn import _array_core as np

from morie.fn.cv1gn import cv1_genomic


def _mean(xs):
    s = 0.0
    for v in xs:
        s += v
    return s / len(xs)


def _mean_center(xs):
    m = _mean(xs)
    return [v - m for v in xs]


def _pearson_corr(a, b):
    na = _mean_center(a)
    nb = _mean_center(b)
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(len(a)):
        num += na[i] * nb[i]
        da += na[i] * na[i]
        db += nb[i] * nb[i]
    if da == 0.0 or db == 0.0:
        return float("nan")
    return num / math.sqrt(da * db)


def _mse(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s / len(a)


def test_cv1gn_basic():
    """Test basic functionality against the documented formula."""
    rng_y = np.random.default_rng(43).normal(0, 1, 100)
    y = [float(v) for v in rng_y]
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    X = [[float(e) for e in row] for row in markers]
    n_folds = 5

    result = cv1_genomic(y, X, n_folds)

    assert hasattr(result, "estimate")
    assert hasattr(result, "pa")
    assert hasattr(result, "mse")
    assert hasattr(result, "y_hat")
    assert hasattr(result, "pa_fold")
    assert hasattr(result, "fold")

    expected_pa = _pearson_corr(y, list(result.y_hat))
    expected_mse = _mse(y, list(result.y_hat))

    assert math.isclose(result.pa, expected_pa, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(result.estimate, expected_pa, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(result.mse, expected_mse, rel_tol=1e-9, abs_tol=1e-9)

    assert len(result.y_hat) == len(y)
    assert len(result.fold) == len(y)

    expected_pa_fold = []
    K = n_folds
    for f in range(K):
        te = [i for i in range(len(y)) if result.fold[i] == f]
        if len(te) > 1:
            yt = [y[i] for i in te]
            yh = [result.y_hat[i] for i in te]
            expected_pa_fold.append(_pearson_corr(yt, yh))
        else:
            expected_pa_fold.append(float("nan"))
    assert len(result.pa_fold) == K
    for got, exp in zip(result.pa_fold, expected_pa_fold):
        if math.isnan(exp):
            assert math.isnan(got)
        else:
            assert math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-9)


def test_cv1gn_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43).normal(0, 1, 100)
    y = [float(v) for v in rng_y]
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    X = [[float(e) for e in row] for row in markers]

    result = cv1_genomic(y, X, 5)
    assert hasattr(result, "y_hat")
    assert len(result.y_hat) == len(y)
