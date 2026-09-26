"""glmmre is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmre import glmmre


def test_glmmre_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmre()
