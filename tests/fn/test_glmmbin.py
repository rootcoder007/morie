"""glmmbin is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmbin import glmmbin


def test_glmmbin_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmbin()
