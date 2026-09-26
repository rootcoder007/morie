"""rnagg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnagg import rnagg


def test_rnagg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnagg()
