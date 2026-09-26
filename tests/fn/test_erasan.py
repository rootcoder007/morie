"""erasan is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.erasan import erasan


def test_erasan_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        erasan()
