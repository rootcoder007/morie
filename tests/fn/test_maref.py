"""maref is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maref import maref


def test_maref_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maref()
