"""hyspi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyspi import hyspi


def test_hyspi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyspi()
