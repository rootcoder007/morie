"""glmmsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmsim import glmmsim


def test_glmmsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmsim()
