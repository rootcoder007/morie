"""hyfpl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyfpl import hyfpl


def test_hyfpl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyfpl()
