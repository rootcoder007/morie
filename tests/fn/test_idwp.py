"""idwp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwp import idwp


def test_idwp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwp()
