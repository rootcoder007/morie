"""bookadvanced_elementsofstatisticallearning9e1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning9e1 import (
    bookadvanced_elementsofstatisticallearning_chapter_9_equation_1,
)


def test_bookadvanced_elementsofstatisticallearning9e1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_9_equation_1(x=None)
