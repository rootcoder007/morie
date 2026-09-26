"""hedderich8e43 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hedderich8e43 import hedderich_chapter_8_equation_43


def test_hedderich8e43_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hedderich_chapter_8_equation_43(x=None)
