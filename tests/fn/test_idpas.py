"""idpas is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpas import idpas


def test_idpas_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpas()
