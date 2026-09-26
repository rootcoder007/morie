"""wilcox7e8 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e8 import wilcox_chapter_7_equation_8


def test_wilcox7e8_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_8(x=None)
