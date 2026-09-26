"""lusim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.lusim import lusim


def test_lusim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lusim()
