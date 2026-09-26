"""gdtfr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdtfr import gdtfr


def test_gdtfr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdtfr()
