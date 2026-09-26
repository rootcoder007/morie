"""clcon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clcon import clcon


def test_clcon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clcon()
