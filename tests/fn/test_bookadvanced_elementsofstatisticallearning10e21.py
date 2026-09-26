"""bookadvanced_elementsofstatisticallearning10e21 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning10e21 import (
    bookadvanced_elementsofstatisticallearning_chapter_10_equation_21,
)


def test_bookadvanced_elementsofstatisticallearning10e21_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_10_equation_21(x=None)
