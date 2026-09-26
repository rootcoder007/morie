"""manh4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.manh4 import manh4


def test_manh4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        manh4()
