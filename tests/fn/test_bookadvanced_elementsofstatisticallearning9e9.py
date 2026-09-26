"""bookadvanced_elementsofstatisticallearning9e9 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning9e9 import (
    bookadvanced_elementsofstatisticallearning_chapter_9_equation_9,
)


def test_bookadvanced_elementsofstatisticallearning9e9_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_9_equation_9(x=None)
