"""agdvm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agdvm import agdvm


def test_agdvm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agdvm()
