"""cb15e4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cb15e4 import cb_chapter_15_equation_4


def test_cb15e4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cb_chapter_15_equation_4(x=None)
