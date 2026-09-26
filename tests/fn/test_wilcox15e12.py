"""wilcox15e12 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox15e12 import wilcox_chapter_15_equation_12


def test_wilcox15e12_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_15_equation_12(x=None)
