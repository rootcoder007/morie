"""gdcst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdcst import gdcst


def test_gdcst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdcst()
