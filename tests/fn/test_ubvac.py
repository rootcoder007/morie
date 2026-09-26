"""ubvac is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubvac import ubvac


def test_ubvac_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubvac()
