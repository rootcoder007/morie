"""rsusc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rsusc import rsusc


def test_rsusc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rsusc()
