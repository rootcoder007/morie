"""chlrng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlrng import chlrng


def test_chlrng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlrng()
