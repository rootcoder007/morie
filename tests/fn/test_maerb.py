"""maerb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maerb import maerb


def test_maerb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maerb()
