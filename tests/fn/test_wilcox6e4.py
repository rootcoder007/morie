"""wilcox6e4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox6e4 import wilcox_chapter_6_equation_4


def test_wilcox6e4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_6_equation_4(x=None)
