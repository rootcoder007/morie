"""wilcox7e36 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e36 import wilcox_chapter_7_equation_36


def test_wilcox7e36_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_36(x=None)
