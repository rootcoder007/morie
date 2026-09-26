"""gcacd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcacd import gcacd


def test_gcacd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcacd()
