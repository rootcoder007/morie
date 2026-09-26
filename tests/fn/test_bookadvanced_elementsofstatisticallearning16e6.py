"""bookadvanced_elementsofstatisticallearning16e6 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning16e6 import (
    bookadvanced_elementsofstatisticallearning_chapter_16_equation_6,
)


def test_bookadvanced_elementsofstatisticallearning16e6_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_16_equation_6(x=None)
