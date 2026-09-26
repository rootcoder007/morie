"""shYa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.shYa import shunting_yard


def test_shYa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        shunting_yard(tokens=None)
