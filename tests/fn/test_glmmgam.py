"""glmmgam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmgam import glmmgam


def test_glmmgam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmgam()
