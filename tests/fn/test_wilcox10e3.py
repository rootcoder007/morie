"""wilcox10e3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox10e3 import wilcox_chapter_10_equation_3


def test_wilcox10e3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_10_equation_3(x=None)
