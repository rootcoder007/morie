"""gdvtr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdvtr import gdvtr


def test_gdvtr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdvtr()
