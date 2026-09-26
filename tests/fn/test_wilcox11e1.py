"""wilcox11e1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox11e1 import wilcox_chapter_11_equation_1


def test_wilcox11e1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_11_equation_1(x=None)
