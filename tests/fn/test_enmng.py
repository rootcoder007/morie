"""enmng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enmng import enmng


def test_enmng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enmng()
