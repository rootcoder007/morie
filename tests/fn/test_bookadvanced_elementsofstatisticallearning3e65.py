"""bookadvanced_elementsofstatisticallearning3e65 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning3e65 import (
    bookadvanced_elementsofstatisticallearning_chapter_3_equation_65,
)


def test_bookadvanced_elementsofstatisticallearning3e65_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_3_equation_65(x=None)
