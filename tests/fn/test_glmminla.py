"""glmminla is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmminla import glmminla


def test_glmminla_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmminla()
