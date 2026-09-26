"""idpnm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpnm import idpnm


def test_idpnm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpnm()
