"""sdmspil is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdmspil import sdmspil


def test_sdmspil_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdmspil(indirect=None, total=None)
