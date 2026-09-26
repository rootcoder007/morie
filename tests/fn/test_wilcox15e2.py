"""wilcox15e2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox15e2 import wilcox_chapter_15_equation_2


def test_wilcox15e2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_15_equation_2(x=None)
