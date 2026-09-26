"""wilcox14e17 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox14e17 import wilcox_chapter_14_equation_17


def test_wilcox14e17_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_14_equation_17(x=None)
