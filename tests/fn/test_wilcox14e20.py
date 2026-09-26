"""wilcox14e20 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e20 import wilcox_chapter_14_equation_20


def test_wilcox14e20_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_20(x=None)
