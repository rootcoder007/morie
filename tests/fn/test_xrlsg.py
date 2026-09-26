"""xrlsg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrlsg import lisa_getis


def test_xrlsg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lisa_getis(data=None)
