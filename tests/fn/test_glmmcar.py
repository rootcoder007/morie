"""glmmcar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmcar import glmmcar


def test_glmmcar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmcar()
