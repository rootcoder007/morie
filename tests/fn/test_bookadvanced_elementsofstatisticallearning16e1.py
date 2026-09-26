"""bookadvanced_elementsofstatisticallearning16e1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning16e1 import (
    bookadvanced_elementsofstatisticallearning_chapter_16_equation_1,
)


def test_bookadvanced_elementsofstatisticallearning16e1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_16_equation_1(x=None)
