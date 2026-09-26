"""sogrv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sogrv import sogrv


def test_sogrv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sogrv()
