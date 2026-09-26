"""svord is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svord import svord


def test_svord_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svord()
