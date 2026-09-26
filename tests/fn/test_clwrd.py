"""clwrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clwrd import clwrd


def test_clwrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clwrd()
