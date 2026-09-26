"""afslp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afslp import afslp


def test_afslp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afslp()
