"""wlmxe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlmxe import wlmxe


def test_wlmxe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlmxe()
