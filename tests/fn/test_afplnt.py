"""afplnt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afplnt import afplnt


def test_afplnt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afplnt()
