"""idpam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpam import idpam


def test_idpam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpam()
