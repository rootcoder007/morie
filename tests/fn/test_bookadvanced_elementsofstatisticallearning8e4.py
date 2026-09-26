"""bookadvanced_elementsofstatisticallearning8e4 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning8e4 import (
    bookadvanced_elementsofstatisticallearning_chapter_8_equation_4,
)


def test_bookadvanced_elementsofstatisticallearning8e4_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_8_equation_4(x=None)
