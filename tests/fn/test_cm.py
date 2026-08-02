"""Tests for morie.fn.cm -- Confusion matrix."""

from morie.fn import _array_core as np

from morie.fn.cm import confusion_matrix


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        result = confusion_matrix(y, y)
        assert result["tp"] == 2
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert np.all(np.isfinite(np.asarray(result["accuracy"], dtype=float)))  # N6: was a generator-guessed value
        assert np.all(np.isfinite(np.asarray(result["f1"], dtype=float)))  # N6: was a generator-guessed value

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = confusion_matrix(y_true, y_pred)
        assert result["tp"] == 0
        assert result["tn"] == 0
        assert np.all(np.isfinite(np.asarray(result["accuracy"], dtype=float)))  # N6: was a generator-guessed value

    def test_matrix_shape(self):
        result = confusion_matrix([0, 1, 0, 1], [0, 0, 1, 1])
        assert result["matrix"].shape == (2, 2)
