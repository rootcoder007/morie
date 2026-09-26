"""wilcox15e9 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox15e9 import wilcox_chapter_15_equation_9


def test_wilcox15e9_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_15_equation_9(x=None)
