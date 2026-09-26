"""bookadvanced_elementsofstatisticallearning9e3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning9e3 import (
    bookadvanced_elementsofstatisticallearning_chapter_9_equation_3,
)


def test_bookadvanced_elementsofstatisticallearning9e3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_9_equation_3(x=None)
