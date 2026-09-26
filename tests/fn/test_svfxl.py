"""svfxl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svfxl import svfxl


def test_svfxl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svfxl()
