"""wilcox7e31 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e31 import wilcox_chapter_7_equation_31


def test_wilcox7e31_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_31(x=None)
