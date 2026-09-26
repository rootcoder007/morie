"""afslp2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afslp2 import afslp2


def test_afslp2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afslp2()
