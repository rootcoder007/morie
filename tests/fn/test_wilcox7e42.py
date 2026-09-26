"""wilcox7e42 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e42 import wilcox_chapter_7_equation_42


def test_wilcox7e42_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_42(x=None)
