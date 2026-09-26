"""wilcox6e18 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox6e18 import wilcox_chapter_6_equation_18


def test_wilcox6e18_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_6_equation_18(x=None)
