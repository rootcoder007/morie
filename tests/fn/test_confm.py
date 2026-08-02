"""Tests for confm.confusion_matrix_metrics."""

from morie.fn import _array_core as np
import pytest

from morie.fn.confm import confusion_matrix_metrics


def test_confm_matches_a_hand_computed_table():
    """y_true = (0,0,1,1,1,2), y_pred = (0,1,1,1,2,2). The table, the
    per-class precision/recall and every F1 can be written down: all three
    F1 scores come out exactly 2/3."""
    r = confusion_matrix_metrics([0, 0, 1, 1, 1, 2], [0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(np.asarray(r["confusion_matrix"]), [[1, 1, 0], [0, 2, 1], [0, 0, 1]])
    assert float(r["accuracy"]) == pytest.approx(4 / 6, rel=1e-12)
    np.testing.assert_allclose(r["precision"], [1.0, 2 / 3, 1 / 2], atol=1e-12)
    np.testing.assert_allclose(r["recall"], [1 / 2, 2 / 3, 1.0], atol=1e-12)
    np.testing.assert_allclose(r["f1"], [2 / 3, 2 / 3, 2 / 3], atol=1e-12)
    assert float(r["macro_f1"]) == pytest.approx(2 / 3, rel=1e-12)


def test_confm_perfect_prediction_is_all_ones():
    y = [0, 1, 2, 1, 0]
    r = confusion_matrix_metrics(y, y)
    assert float(r["accuracy"]) == 1.0
    np.testing.assert_allclose(r["precision"], 1.0, atol=1e-12)
    np.testing.assert_allclose(r["recall"], 1.0, atol=1e-12)
    np.testing.assert_allclose(r["f1"], 1.0, atol=1e-12)


def test_confm_f1_is_the_harmonic_mean_not_the_arithmetic():
    """With precision 1 and recall 1/2, F1 = 2/3, not 3/4 -- the harmonic
    mean punishes imbalance (van Rijsbergen 1979, Ch. 7)."""
    r = confusion_matrix_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    i = list(r["labels"]).index(1)
    assert float(np.asarray(r["precision"])[i]) == pytest.approx(1.0, abs=1e-12)
    assert float(np.asarray(r["recall"])[i]) == pytest.approx(0.5, abs=1e-12)
    assert float(np.asarray(r["f1"])[i]) == pytest.approx(2 / 3, rel=1e-12)


def test_confm_explicit_labels_keep_absent_classes():
    r = confusion_matrix_metrics([0, 0], [0, 0], labels=[0, 1])
    cm = np.asarray(r["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm[1].sum() == 0
