"""wilcox14e22 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e22 import wilcox_chapter_14_equation_22


def test_wilcox14e22_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_22(x=None)
