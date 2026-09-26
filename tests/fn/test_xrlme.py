"""xrlme is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrlme import lm_error


def test_xrlme_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        lm_error(data=None)
