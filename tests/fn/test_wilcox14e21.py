"""wilcox14e21 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e21 import wilcox_chapter_14_equation_21


def test_wilcox14e21_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_21(x=None)
