"""bookadvanced_elementsofstatisticallearning3e46 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning3e46 import (
    bookadvanced_elementsofstatisticallearning_chapter_3_equation_46,
)


def test_bookadvanced_elementsofstatisticallearning3e46_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_3_equation_46(x=None)
