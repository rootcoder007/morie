"""glmmdiag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmdiag import glmmdiag


def test_glmmdiag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmdiag()
