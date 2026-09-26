"""trcng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trcng import trcng


def test_trcng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trcng()
