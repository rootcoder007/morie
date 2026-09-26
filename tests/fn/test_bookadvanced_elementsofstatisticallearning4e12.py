"""bookadvanced_elementsofstatisticallearning4e12 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning4e12 import (
    bookadvanced_elementsofstatisticallearning_chapter_4_equation_12,
)


def test_bookadvanced_elementsofstatisticallearning4e12_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_4_equation_12(x=None)
