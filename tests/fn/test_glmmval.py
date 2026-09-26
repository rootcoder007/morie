"""glmmval is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmval import glmmval


def test_glmmval_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmval()
