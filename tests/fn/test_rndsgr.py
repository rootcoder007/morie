"""rndsgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rndsgr import rndsgr


def test_rndsgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rndsgr()
