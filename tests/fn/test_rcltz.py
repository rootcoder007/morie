"""rcltz is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcltz import rcltz


def test_rcltz_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcltz()
