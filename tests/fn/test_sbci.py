"""sbci is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbci import sbci


def test_sbci_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbci()
