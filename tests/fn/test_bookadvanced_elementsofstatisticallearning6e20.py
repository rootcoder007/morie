"""bookadvanced_elementsofstatisticallearning6e20 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning6e20 import (
    bookadvanced_elementsofstatisticallearning_chapter_6_equation_20,
)


def test_bookadvanced_elementsofstatisticallearning6e20_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_6_equation_20(x=None)
