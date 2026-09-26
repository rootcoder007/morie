"""idpfa is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpfa import idpfa


def test_idpfa_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpfa()
