"""zxbet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxbet import betti_numbers


def test_zxbet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        betti_numbers(data=None)
