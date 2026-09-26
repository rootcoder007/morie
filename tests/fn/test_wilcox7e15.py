"""wilcox7e15 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e15 import wilcox_chapter_7_equation_15


def test_wilcox7e15_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_15(x=None)
