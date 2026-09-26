"""vtcon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtcon import vtcon


def test_vtcon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtcon()
