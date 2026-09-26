"""aftrr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aftrr import aftrr


def test_aftrr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aftrr()
