"""Tests for morie.fn.holo_v -- violin plot."""

from morie.fn import _array_core as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
from morie.fn import _frame_core as pd

from morie.fn.holo_v import holo_violin


class TestHoloViolin:
    def test_returns_figure(self):
        df = pd.DataFrame({"x": np.random.default_rng(0).normal(size=50)})
        fig = holo_violin(df, "x")
        assert fig is not None
        assert type(fig).__name__ == "Figure"
        plt.close(fig)

    def test_grouped(self):
        df = pd.DataFrame({"x": np.arange(20, dtype=float), "g": ["a"] * 10 + ["b"] * 10})
        fig = holo_violin(df, "x", group="g")
        assert fig is not None
        plt.close(fig)
