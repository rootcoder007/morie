"""mtcon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtcon import mtcon


def test_mtcon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtcon()
