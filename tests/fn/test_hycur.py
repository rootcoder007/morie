"""hycur is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hycur import hycur


def test_hycur_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hycur()
