"""wilcox13e5 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox13e5 import wilcox_chapter_13_equation_5


def test_wilcox13e5_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_13_equation_5(x=None)
