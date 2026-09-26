"""wilcox9e6 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox9e6 import wilcox_chapter_9_equation_6


def test_wilcox9e6_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_9_equation_6(x=None)
