"""hyfdr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyfdr import hyfdr


def test_hyfdr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyfdr()
