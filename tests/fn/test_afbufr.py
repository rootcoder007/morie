"""afbufr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afbufr import afbufr


def test_afbufr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afbufr()
