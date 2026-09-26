"""wilcox14e26 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e26 import wilcox_chapter_14_equation_26


def test_wilcox14e26_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_26(x=None)
