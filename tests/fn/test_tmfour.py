"""tmfour is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmfour import tmfour


def test_tmfour_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmfour()
