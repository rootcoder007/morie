"""srbnb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srbnb import srbnb


def test_srbnb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srbnb()
