"""wilcox14e16 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e16 import wilcox_chapter_14_equation_16


def test_wilcox14e16_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_16(x=None)
