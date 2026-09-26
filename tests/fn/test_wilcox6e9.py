"""wilcox6e9 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox6e9 import wilcox_chapter_6_equation_9


def test_wilcox6e9_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_6_equation_9(x=None)
