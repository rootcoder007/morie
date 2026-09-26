"""wilcox4e14 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox4e14 import wilcox_chapter_4_equation_14


def test_wilcox4e14_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_4_equation_14(x=None)
