"""idpir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpir import idpir


def test_idpir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpir()
