"""wilcox5e5 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox5e5 import wilcox_chapter_5_equation_5


def test_wilcox5e5_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_5_equation_5(x=None)
