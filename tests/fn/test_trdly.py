"""trdly is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trdly import trdly


def test_trdly_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trdly()
