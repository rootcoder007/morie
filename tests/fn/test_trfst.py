"""trfst is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trfst import trfst


def test_trfst_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trfst()
