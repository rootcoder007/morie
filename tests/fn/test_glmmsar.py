"""glmmsar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.glmmsar import glmmsar


def test_glmmsar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        glmmsar()
