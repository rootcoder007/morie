"""sbmnl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbmnl import sbmnl


def test_sbmnl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbmnl()
