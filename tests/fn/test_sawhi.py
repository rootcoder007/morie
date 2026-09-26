"""sawhi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawhi import sawhi


def test_sawhi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawhi()
