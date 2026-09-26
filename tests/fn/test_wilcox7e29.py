"""wilcox7e29 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e29 import wilcox_chapter_7_equation_29


def test_wilcox7e29_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_29(x=None)
