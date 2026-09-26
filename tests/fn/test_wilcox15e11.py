"""wilcox15e11 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox15e11 import wilcox_chapter_15_equation_11


def test_wilcox15e11_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_15_equation_11(x=None)
