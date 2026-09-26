"""wilcox12e7 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox12e7 import wilcox_chapter_12_equation_7


def test_wilcox12e7_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_12_equation_7(x=None)
