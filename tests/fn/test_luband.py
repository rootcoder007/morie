"""luband is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.luband import luband


def test_luband_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        luband()
