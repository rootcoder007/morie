"""wilcox3e15 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox3e15 import wilcox_chapter_3_equation_15


def test_wilcox3e15_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_3_equation_15(x=None)
