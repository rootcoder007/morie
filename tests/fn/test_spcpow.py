"""spcpow is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcpow import spcpow


def test_spcpow_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcpow()
