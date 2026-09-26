"""wlhme is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlhme import wlhme


def test_wlhme_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlhme()
