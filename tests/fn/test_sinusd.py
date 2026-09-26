"""sinusd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sinusd import sinusd


def test_sinusd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sinusd()
