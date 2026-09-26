"""wilcox3e7 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox3e7 import wilcox_chapter_3_equation_7


def test_wilcox3e7_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_3_equation_7(x=None)
