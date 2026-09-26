"""idpvt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpvt import idpvt


def test_idpvt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpvt()
