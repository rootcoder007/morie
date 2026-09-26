"""bookadvanced_elementsofstatisticallearning16e7 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning16e7 import (
    bookadvanced_elementsofstatisticallearning_chapter_16_equation_7,
)


def test_bookadvanced_elementsofstatisticallearning16e7_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_16_equation_7(x=None)
