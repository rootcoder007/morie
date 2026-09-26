"""idpfl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpfl import idpfl


def test_idpfl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpfl()
