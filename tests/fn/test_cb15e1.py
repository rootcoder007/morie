"""cb15e1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cb15e1 import cb_chapter_15_equation_1


def test_cb15e1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cb_chapter_15_equation_1(x=None)
