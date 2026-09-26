"""afaqua is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afaqua import afaqua


def test_afaqua_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afaqua()
