"""Tests for svmhg.svm_hinge_primal."""

from morie.fn import _array_core as np
import pytest

from morie.fn.svmhg import svm_hinge_primal


def _separable(seed=0, n=200, margin=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X[y == 1] += margin / np.sqrt(2)
    X[y == 0] -= margin / np.sqrt(2)
    return X, y


def test_svmhg_separates_separable_data():
    X, y = _separable()
    r = svm_hinge_primal(X, y, C=10.0)
    assert float(r["train_accuracy"]) >= 0.99
    # The separating direction is (1, 1): both weights positive.
    w = np.asarray(r["weights"], dtype=float)
    assert w[0] > 0 and w[1] > 0


def test_svmhg_labels_map_back_to_the_original_classes():
    X, y = _separable(seed=1)
    r = svm_hinge_primal(X, np.where(y == 1, "pos", "neg"), C=1.0)
    assert sorted(r["classes"]) == ["neg", "pos"]


def test_svmhg_small_C_regularises_the_weights():
    """C multiplies the hinge term, so shrinking C shrinks ||w|| -- the
    text-book direction of the trade-off (ESL Sec. 12.2)."""
    X, y = _separable(seed=2)
    w_hi = np.linalg.norm(svm_hinge_primal(X, y, C=10.0)["weights"])
    w_lo = np.linalg.norm(svm_hinge_primal(X, y, C=1e-3)["weights"])
    assert w_lo < w_hi


def test_svmhg_rejects_nonbinary_y():
    X, _ = _separable()
    with pytest.raises(ValueError, match="binary"):
        svm_hinge_primal(X, np.arange(X.shape[0]) % 3)
