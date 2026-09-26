"""afyld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afyld import afyld


def test_afyld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afyld()
