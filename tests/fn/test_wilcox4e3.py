"""wilcox4e3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox4e3 import wilcox_chapter_4_equation_3


def test_wilcox4e3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_4_equation_3(x=None)
