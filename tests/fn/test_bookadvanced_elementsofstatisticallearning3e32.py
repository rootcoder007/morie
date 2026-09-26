"""bookadvanced_elementsofstatisticallearning3e32 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning3e32 import (
    bookadvanced_elementsofstatisticallearning_chapter_3_equation_32,
)


def test_bookadvanced_elementsofstatisticallearning3e32_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_3_equation_32(x=None)
