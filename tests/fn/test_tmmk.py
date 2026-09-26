"""tmmk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmmk import tmmk


def test_tmmk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmmk()
