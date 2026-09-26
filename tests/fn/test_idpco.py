"""idpco is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpco import idpco


def test_idpco_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpco()
