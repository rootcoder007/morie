"""gdcbr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdcbr import gdcbr


def test_gdcbr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdcbr()
