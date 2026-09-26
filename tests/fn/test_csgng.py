"""csgng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csgng import csgng


def test_csgng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csgng()
