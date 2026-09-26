"""tsscn2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tsscn2 import tsscn2


def test_tsscn2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tsscn2()
