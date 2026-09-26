"""idpmk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpmk import idpmk


def test_idpmk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpmk()
