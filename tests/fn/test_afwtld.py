"""afwtld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afwtld import afwtld


def test_afwtld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afwtld()
