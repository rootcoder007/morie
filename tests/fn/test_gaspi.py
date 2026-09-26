"""gaspi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gaspi import gaspi


def test_gaspi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gaspi()
