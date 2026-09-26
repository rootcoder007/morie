"""chlpow is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlpow import chlpow


def test_chlpow_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlpow()
