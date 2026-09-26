"""wilcox6e5 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox6e5 import wilcox_chapter_6_equation_5


def test_wilcox6e5_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_6_equation_5(x=None)
