"""idpw1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpw1 import idpw1


def test_idpw1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpw1()
