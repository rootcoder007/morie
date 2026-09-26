"""bookadvanced_elementsofstatisticallearning4e40 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning4e40 import (
    bookadvanced_elementsofstatisticallearning_chapter_4_equation_40,
)


def test_bookadvanced_elementsofstatisticallearning4e40_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_4_equation_40(x=None)
