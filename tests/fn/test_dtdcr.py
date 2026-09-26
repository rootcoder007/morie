"""dtdcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtdcr import dtdcr


def test_dtdcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtdcr()
