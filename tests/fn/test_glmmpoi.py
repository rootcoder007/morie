"""glmmpoi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmpoi import glmmpoi


def test_glmmpoi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmpoi()
