"""ghbar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghbar import ghbar


def test_ghbar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghbar()
