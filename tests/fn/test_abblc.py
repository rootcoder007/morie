"""abblc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.abblc import abblc


def test_abblc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        abblc()
