"""hsnsh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hsnsh import hsnsh


def test_hsnsh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hsnsh()
