"""clshr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clshr import clshr


def test_clshr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clshr()
