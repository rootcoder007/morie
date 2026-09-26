"""mamng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mamng import mamng


def test_mamng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mamng()
