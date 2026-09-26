"""bookadvanced_elementsofstatisticallearning11e18 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.bookadvanced_elementsofstatisticallearning11e18 import (
    bookadvanced_elementsofstatisticallearning_chapter_11_equation_18,
)


def test_bookadvanced_elementsofstatisticallearning11e18_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        bookadvanced_elementsofstatisticallearning_chapter_11_equation_18(x=None)
