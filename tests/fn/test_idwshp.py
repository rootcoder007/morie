"""idwshp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwshp import idwshp


def test_idwshp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwshp()
