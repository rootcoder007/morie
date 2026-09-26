"""afslk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afslk import afslk


def test_afslk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afslk()
