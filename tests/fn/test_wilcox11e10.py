"""wilcox11e10 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox11e10 import wilcox_chapter_11_equation_10


def test_wilcox11e10_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_11_equation_10(x=None)
