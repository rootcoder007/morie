"""zsrng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsrng import random_nongauss


def test_zsrng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        random_nongauss(data=None)
