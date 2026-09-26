"""ghmlv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghmlv import ghmlv


def test_ghmlv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghmlv()
