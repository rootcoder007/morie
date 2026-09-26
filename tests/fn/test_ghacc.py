"""ghacc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghacc import ghacc


def test_ghacc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghacc()
