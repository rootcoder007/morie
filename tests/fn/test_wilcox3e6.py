"""wilcox3e6 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox3e6 import wilcox_chapter_3_equation_6


def test_wilcox3e6_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_3_equation_6(x=None)
