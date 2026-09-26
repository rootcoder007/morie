"""wilcox7e30 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e30 import wilcox_chapter_7_equation_30


def test_wilcox7e30_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_30(x=None)
