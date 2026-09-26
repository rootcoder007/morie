"""bookadvanced_elementsofstatisticallearning4e3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning4e3 import (
    bookadvanced_elementsofstatisticallearning_chapter_4_equation_3,
)


def test_bookadvanced_elementsofstatisticallearning4e3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_4_equation_3(x=None)
