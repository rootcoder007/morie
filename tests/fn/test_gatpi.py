"""gatpi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gatpi import gatpi


def test_gatpi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gatpi()
