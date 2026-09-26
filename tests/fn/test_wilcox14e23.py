"""wilcox14e23 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e23 import wilcox_chapter_14_equation_23


def test_wilcox14e23_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_23(x=None)
