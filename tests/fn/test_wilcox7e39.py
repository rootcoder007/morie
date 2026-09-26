"""wilcox7e39 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wilcox7e39 import wilcox_chapter_7_equation_39


def test_wilcox7e39_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wilcox_chapter_7_equation_39(x=None)
