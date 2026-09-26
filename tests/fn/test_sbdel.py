"""sbdel is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbdel import sbdel


def test_sbdel_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbdel()
