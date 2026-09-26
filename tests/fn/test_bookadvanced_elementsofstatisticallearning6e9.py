"""bookadvanced_elementsofstatisticallearning6e9 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning6e9 import (
    bookadvanced_elementsofstatisticallearning_chapter_6_equation_9,
)


def test_bookadvanced_elementsofstatisticallearning6e9_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_6_equation_9(x=None)
