"""idwlcl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwlcl import idwlcl


def test_idwlcl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwlcl()
