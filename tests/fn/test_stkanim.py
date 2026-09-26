"""stkanim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stkanim import stkanim


def test_stkanim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stkanim()
