"""idwbst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.idwbst import idwbst


def test_idwbst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        idwbst()
