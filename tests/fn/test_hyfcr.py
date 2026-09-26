"""hyfcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyfcr import hyfcr


def test_hyfcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyfcr()
