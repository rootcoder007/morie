"""wilcox14e27 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e27 import wilcox_chapter_14_equation_27


def test_wilcox14e27_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_27(x=None)
