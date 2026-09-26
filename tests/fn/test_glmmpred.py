"""glmmpred is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmpred import glmmpred


def test_glmmpred_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmpred()
