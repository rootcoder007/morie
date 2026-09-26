"""afwed is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afwed import afwed


def test_afwed_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afwed()
