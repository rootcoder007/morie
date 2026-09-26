"""idpw2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpw2 import idpw2


def test_idpw2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpw2()
