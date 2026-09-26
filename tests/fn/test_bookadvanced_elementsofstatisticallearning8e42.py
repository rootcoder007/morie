"""bookadvanced_elementsofstatisticallearning8e42 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning8e42 import (
    bookadvanced_elementsofstatisticallearning_chapter_8_equation_42,
)


def test_bookadvanced_elementsofstatisticallearning8e42_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_8_equation_42(x=None)
