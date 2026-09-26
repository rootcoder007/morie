"""enhbt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enhbt import enhbt


def test_enhbt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enhbt()
