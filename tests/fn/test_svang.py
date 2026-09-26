"""svang is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svang import svang


def test_svang_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svang()
