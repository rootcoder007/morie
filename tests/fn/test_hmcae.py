"""Tests for hmcae.geron_convolutional_autoencoder."""

from morie.fn import _array_core as np

from morie.fn.hmcae import geron_convolutional_autoencoder


def _img():
    return [[float((3 * i + 5 * j) % 7) + 0.25 * i for j in range(4)]
            for i in range(4)]


def _patch_floor(X, P, F):
    """Eckart-Young floor for a bias-free rank-F linear code: the mean
    squared error per pixel is at least the sum of the P*P - F smallest
    eigenvalues of the uncentred patch second-moment matrix over P*P."""
    H, W = len(X), len(X[0])
    ps = [[X[i + u][j + v] for u in range(P) for v in range(P)]
          for i in range(0, H, P) for j in range(0, W, P)]
    N, d = len(ps), P * P
    M = [[sum(p[a] * p[b] for p in ps) / N for b in range(d)] for a in range(d)]
    ev = sorted(np.linalg.eigvalsh(np.array(M)).tolist())
    return sum(ev[:d - F]) / d


def test_hmcae_basic():
    """A one-filter bottleneck on 2x2 patches: no epoch can beat the
    Eckart-Young floor, and training lowers the loss from its start."""
    X = _img()
    result = geron_convolutional_autoencoder(X, filters=1, epochs=400, lr=0.01, seed=3)
    assert isinstance(result, dict)
    floor = _patch_floor(X, 2, 1)
    assert floor > 0.0
    assert min(result["loss_history"]) >= floor * (1.0 - 1e-9)
    assert result["final_loss"] < result["loss_history"][0]
    assert result["compression_ratio"] == 4.0
    assert result["code_shape"] == (1, 2, 2)


def test_hmcae_edge():
    """Invalid shapes are refused rather than silently cropped."""
    import pytest
    with pytest.raises(ValueError):
        geron_convolutional_autoencoder([[1.0, 2.0, 3.0]] * 3, filters=1)
    with pytest.raises(ValueError):
        geron_convolutional_autoencoder(_img(), filters=0)


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmcae as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
