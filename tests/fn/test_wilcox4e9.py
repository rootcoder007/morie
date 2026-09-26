"""wilcox4e9 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox4e9 import wilcox_chapter_4_equation_9


def test_wilcox4e9_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_4_equation_9(x=None)
