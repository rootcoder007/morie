"""bookadvanced_elementsofstatisticallearning8e6 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning8e6 import (
    bookadvanced_elementsofstatisticallearning_chapter_8_equation_6,
)


def test_bookadvanced_elementsofstatisticallearning8e6_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_8_equation_6(x=None)
