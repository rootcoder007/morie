"""aglpi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aglpi import aglpi


def test_aglpi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aglpi()
