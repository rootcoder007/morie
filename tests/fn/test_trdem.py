"""trdem is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trdem import trdem


def test_trdem_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trdem()
