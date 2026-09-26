"""bookadvanced_elementsofstatisticallearning11e13 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning11e13 import (
    bookadvanced_elementsofstatisticallearning_chapter_11_equation_13,
)


def test_bookadvanced_elementsofstatisticallearning11e13_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_11_equation_13(x=None)
