"""wilcox13e2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox13e2 import wilcox_chapter_13_equation_2


def test_wilcox13e2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_13_equation_2(x=None)
