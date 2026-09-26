"""abmcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abmcr import abmcr


def test_abmcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abmcr()
