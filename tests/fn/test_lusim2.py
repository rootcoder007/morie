"""lusim2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lusim2 import lusim2


def test_lusim2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lusim2()
