"""trfry is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trfry import trfry


def test_trfry_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trfry()
