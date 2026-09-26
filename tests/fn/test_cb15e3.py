"""cb15e3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cb15e3 import cb_chapter_15_equation_3


def test_cb15e3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cb_chapter_15_equation_3(x=None)
