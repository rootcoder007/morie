"""wilcox6e24 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox6e24 import wilcox_chapter_6_equation_24


def test_wilcox6e24_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_6_equation_24(x=None)
