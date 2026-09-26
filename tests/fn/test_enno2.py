"""enno2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enno2 import enno2


def test_enno2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enno2()
