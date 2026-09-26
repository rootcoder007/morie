"""wilcox14e18 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e18 import wilcox_chapter_14_equation_18


def test_wilcox14e18_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_18(x=None)
