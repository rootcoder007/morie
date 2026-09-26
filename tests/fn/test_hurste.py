"""hurste is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hurste import hurst_exponent


def test_hurste_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hurst_exponent(y=None)
