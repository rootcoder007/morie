"""hynit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hynit import hynit


def test_hynit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hynit()
