"""afslm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afslm import afslm


def test_afslm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afslm()
