"""trair is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trair import trair


def test_trair_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trair()
