"""abmld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abmld import abmld


def test_abmld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abmld()
