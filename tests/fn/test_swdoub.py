"""swdoub is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swdoub import swdoub


def test_swdoub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swdoub(W=None)
