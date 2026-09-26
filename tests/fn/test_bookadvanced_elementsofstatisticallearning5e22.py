"""bookadvanced_elementsofstatisticallearning5e22 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning5e22 import (
    bookadvanced_elementsofstatisticallearning_chapter_5_equation_22,
)


def test_bookadvanced_elementsofstatisticallearning5e22_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_5_equation_22(x=None)
