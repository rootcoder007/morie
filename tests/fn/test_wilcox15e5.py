"""wilcox15e5 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox15e5 import wilcox_chapter_15_equation_5


def test_wilcox15e5_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_15_equation_5(x=None)
