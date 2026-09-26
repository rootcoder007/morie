"""elpol is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elpol import elpol


def test_elpol_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elpol()
