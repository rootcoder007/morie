"""enwtl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enwtl import enwtl


def test_enwtl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enwtl()
