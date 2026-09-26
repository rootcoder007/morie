"""use_r8e5 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r8e5 import use_r_chapter_8_equation_5


def test_use_r8e5_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_8_equation_5(x=None)
