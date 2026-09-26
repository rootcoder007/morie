"""cb7e1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cb7e1 import cb_chapter_7_equation_1


def test_cb7e1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cb_chapter_7_equation_1(x=None)
