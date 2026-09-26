"""wilcox7e26 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e26 import wilcox_chapter_7_equation_26


def test_wilcox7e26_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_26(x=None)
