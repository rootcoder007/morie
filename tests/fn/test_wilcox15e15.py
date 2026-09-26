"""wilcox15e15 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox15e15 import wilcox_chapter_15_equation_15


def test_wilcox15e15_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_15_equation_15(x=None)
