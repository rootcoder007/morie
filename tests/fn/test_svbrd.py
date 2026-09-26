"""svbrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svbrd import borda_spatial


def test_svbrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        borda_spatial(data=None)
