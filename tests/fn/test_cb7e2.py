"""cb7e2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cb7e2 import cb_chapter_7_equation_2


def test_cb7e2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cb_chapter_7_equation_2(x=None)
