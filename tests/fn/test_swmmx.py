"""swmmx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swmmx import swmmx


def test_swmmx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swmmx(W=None)
