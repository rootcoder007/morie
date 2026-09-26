"""glmmnb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmnb import glmmnb


def test_glmmnb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmnb()
