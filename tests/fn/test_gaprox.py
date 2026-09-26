"""gaprox is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gaprox import gaprox


def test_gaprox_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gaprox()
