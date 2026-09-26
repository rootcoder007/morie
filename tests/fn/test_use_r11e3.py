"""use_r11e3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.use_r11e3 import use_r_chapter_11_equation_3


def test_use_r11e3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        use_r_chapter_11_equation_3(x=None)
