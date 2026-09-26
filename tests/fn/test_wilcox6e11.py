"""wilcox6e11 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox6e11 import wilcox_chapter_6_equation_11


def test_wilcox6e11_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_6_equation_11(x=None)
