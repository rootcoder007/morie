"""wilcox13e8 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox13e8 import wilcox_chapter_13_equation_8


def test_wilcox13e8_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_13_equation_8(x=None)
