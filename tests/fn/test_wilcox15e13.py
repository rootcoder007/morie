"""wilcox15e13 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox15e13 import wilcox_chapter_15_equation_13


def test_wilcox15e13_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_15_equation_13(x=None)
