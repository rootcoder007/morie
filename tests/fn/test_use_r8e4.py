"""use_r8e4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r8e4 import use_r_chapter_8_equation_4


def test_use_r8e4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_8_equation_4(x=None)
