"""wilcox14e24 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e24 import wilcox_chapter_14_equation_24


def test_wilcox14e24_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_24(x=None)
