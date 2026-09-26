"""idppc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idppc import idppc


def test_idppc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idppc()
