"""idpem is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idpem import idpem


def test_idpem_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idpem()
