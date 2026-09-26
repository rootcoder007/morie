"""chlcnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlcnd import chlcnd


def test_chlcnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlcnd()
