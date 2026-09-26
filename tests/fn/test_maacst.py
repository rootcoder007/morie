"""maacst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maacst import maacst


def test_maacst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maacst()
